// BAM record boundary discovery and columnar field decode, on device.
//
// This is a direct translation of `fritillaria-bam/src/blocked.rs`, which is
// the CPU reference and the oracle these kernels are diffed against. Read that
// module first: it explains why the boundary scan can be parallelised at all
// and why speculating on block starts is sound rather than merely lucky.
//
// The short version. Record n+1 starts where record n ends, so a scan is
// inherently serial — fatal on a device, where a chain of 55,000 dependent
// loads costs more than inflating the batch did. But htslib ends a BGZF block
// early rather than splitting a record across two, so a block start is almost
// always a record start. That turns one chain into one chain per block.
//
// Correctness never depends on the guess being right: a block's speculative
// walk is used only once the true chain is proven to arrive at that block's
// start, and a walk from a true boundary is the true walk. Blocks a long ONT
// read runs straight through fall back to walking, in `bam_reconcile`.
//
// Four phases, three of them parallel:
//
//   bam_scan_blocks   one thread per block   speculative walks
//   bam_reconcile     one thread             follow the true chain over blocks
//   bam_emit_offsets  one thread per block   write each record's offset
//   bam_decode_fields one thread per record  write the columns
//
// Loads are byte-wise throughout. BAM records carry no alignment guarantee —
// a record starts wherever the previous one ended — so reinterpreting a
// pointer as `unsigned int*` would be undefined and, on this hardware, wrong
// rather than merely slow.

#define FR_BAM_OK 0u
#define FR_BAM_MALFORMED 1u

// A record's fixed core, excluding the block_size prefix.
#define RECORD_CORE_SIZE 32u
// Bytes from a record's start to its read name: the prefix plus the core.
#define NAME_START 36u

// No record starts in this block: an earlier record covers all of it.
#define ENTRY_NONE 0xFFFFFFFFFFFFFFFFull

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

__device__ __forceinline__ u32 load_u32(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ __forceinline__ u32 load_u16(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8);
}

// Whether the variable-length fields a record declares fit inside the length
// its prefix claims.
//
// This is what makes a speculative walk safe. Starting mid-record yields an
// arbitrary block_size; without this check the walk would stride off into
// nonsense, and with it a bad guess is caught within a record or two, because
// l_read_name, n_cigar_op and l_seq have to add up to exactly the declared
// length. Arithmetic is 64-bit: l_seq is attacker-controlled and (l_seq + 1)
// overflows 32 bits at 0xFFFFFFFF.
__device__ bool fields_fit(const u8 *buf, u64 start, u64 end) {
    u32 name_len = (u32)buf[start + 12];
    if (name_len == 0u) {
        return false; // the NUL terminator is counted, so this is never 0
    }
    u64 n_cigar = (u64)load_u16(buf + start + 16);
    u64 l_seq = (u64)load_u32(buf + start + 20);

    u64 need = (u64)NAME_START + (u64)name_len + 4ull * n_cigar + (l_seq + 1ull) / 2ull + l_seq;
    return start + need <= end;
}

struct Segment {
    u64 land;      // first offset at or past the segment end
    u32 count;     // records starting within the segment
    int complete;  // ended on a boundary rather than inside a clipped record
    int valid;     // every record walked was well formed
};

// Walks records from `from` until reaching `until` or running out of buffer.
//
// `valid = 0` is not necessarily corruption: for a speculative walk from a
// block start that is not a record boundary it is the expected outcome. The
// caller decides which it is. Every iteration advances by at least 36 bytes,
// so this terminates.
__device__ Segment walk_segment(const u8 *buf, u64 len, u64 from, u64 until) {
    Segment s;
    s.land = from;
    s.count = 0u;
    s.complete = 1;
    s.valid = 1;

    u64 cur = from;
    for (;;) {
        if (cur >= until) {
            s.land = cur;
            s.complete = 1;
            return s;
        }
        if (cur + 4ull > len) {
            // A partial length prefix: the tail, not an error.
            s.land = cur;
            s.complete = 0;
            return s;
        }
        u32 block_size = load_u32(buf + cur);
        if (block_size < RECORD_CORE_SIZE) {
            s.land = cur;
            s.valid = 0;
            return s;
        }
        u64 end = cur + 4ull + (u64)block_size;
        if (end > len) {
            // The record is real but the buffer stops inside it.
            s.land = cur;
            s.complete = 0;
            return s;
        }
        if (!fields_fit(buf, cur, end)) {
            s.land = cur;
            s.valid = 0;
            return s;
        }
        s.count += 1u;
        cur = end;
    }
}

// Phase 1: speculate. One thread per block, all independent.
extern "C" __global__ void bam_scan_blocks(const u8 *buf, u64 len, const u64 *block_starts,
                                           int n_blocks, u32 *counts, u64 *lands, u8 *complete,
                                           u8 *valid) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) {
        return;
    }
    Segment s = walk_segment(buf, len, block_starts[b], block_starts[b + 1]);
    counts[b] = s.count;
    lands[b] = s.land;
    complete[b] = (u8)s.complete;
    valid[b] = (u8)s.valid;
}

// Phase 2: reconcile. Serial, but over blocks — hundreds — rather than records.
//
// One thread deliberately. The loop is a dependent chain by nature, and its
// length is the block count, so the parallelism that matters was already taken
// in phase 1. The fallback walk runs only for blocks the chain does not enter
// at their start, which for ultra-long reads is a handful of records.
//
// `totals` is [record count, tail offset, status].
extern "C" __global__ void bam_reconcile(const u8 *buf, u64 len, const u64 *block_starts,
                                         int n_blocks, u64 start, const u32 *spec_counts,
                                         const u64 *spec_lands, const u8 *spec_complete,
                                         const u8 *spec_valid, u64 *entries, u32 *counts,
                                         u32 *first_index, u64 *totals) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    u64 cur = start;
    u64 emitted = 0ull;
    u64 tail = len;
    int stopped = 0;
    u32 status = FR_BAM_OK;

    for (int b = 0; b < n_blocks; ++b) {
        first_index[b] = (u32)emitted;
        counts[b] = 0u;
        entries[b] = ENTRY_NONE;

        if (stopped || cur >= block_starts[b + 1]) {
            // Either the buffer already ran out, or a record that started in an
            // earlier block covers this one entirely.
            continue;
        }

        u32 cnt;
        u64 land;
        int complete;
        int valid;
        if (cur == block_starts[b]) {
            // The true chain arrives exactly at the block start, which proves
            // the speculative walk from there was the true walk.
            cnt = spec_counts[b];
            land = spec_lands[b];
            complete = (int)spec_complete[b];
            valid = (int)spec_valid[b];
        } else {
            Segment s = walk_segment(buf, len, cur, block_starts[b + 1]);
            cnt = s.count;
            land = s.land;
            complete = s.complete;
            valid = s.valid;
        }

        if (!valid) {
            // Reached from a proven boundary, so this is real corruption rather
            // than a bad guess.
            status = FR_BAM_MALFORMED;
            tail = cur;
            stopped = 1;
            continue;
        }

        entries[b] = cur;
        counts[b] = cnt;
        emitted += (u64)cnt;
        cur = land;
        if (!complete) {
            tail = land;
            stopped = 1;
        }
    }

    if (!stopped) {
        tail = cur;
    }
    totals[0] = emitted;
    totals[1] = tail;
    totals[2] = (u64)status;
}

// Phase 3: emit. One thread per block, each writing its own slice of the
// offsets array, so no two threads touch the same element.
extern "C" __global__ void bam_emit_offsets(const u8 *buf, const u64 *entries, const u32 *counts,
                                            const u32 *first_index, int n_blocks,
                                            u64 *record_offsets) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) {
        return;
    }
    u64 cur = entries[b];
    if (cur == ENTRY_NONE) {
        return;
    }
    u32 base = first_index[b];
    u32 n = counts[b];
    for (u32 i = 0u; i < n; ++i) {
        record_offsets[base + i] = cur;
        cur += 4ull + (u64)load_u32(buf + cur);
    }
}

// Phase 4: decode. One thread per record — the embarrassingly parallel part,
// and the only one whose parallelism scales with the record count rather than
// the block count.
//
// Field offsets are emitted rather than left to the caller. They are relative
// to the record start and fit in 32 bits because a record is bounded by its own
// block_size; only the record offsets themselves need 64.
extern "C" __global__ void bam_decode_fields(const u8 *buf, const u64 *record_offsets,
                                             u32 n_records, int *reference_sequence_id, int *position,
                                             unsigned short *flags, u8 *mapping_quality,
                                             u32 *sequence_len, int *mate_reference_sequence_id,
                                             int *mate_position, int *template_length,
                                             u32 *cigar_start, u32 *sequence_start,
                                             u32 *quality_start, u32 *aux_start, u32 *record_end) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_records) {
        return;
    }
    u64 r = record_offsets[i];

    u32 block_size = load_u32(buf + r);
    u32 name_len = (u32)buf[r + 12];
    u32 n_cigar = load_u16(buf + r + 16);
    u32 l_seq = load_u32(buf + r + 20);

    reference_sequence_id[i] = (int)load_u32(buf + r + 4);
    position[i] = (int)load_u32(buf + r + 8);
    mapping_quality[i] = buf[r + 13];
    flags[i] = (unsigned short)load_u16(buf + r + 18);
    sequence_len[i] = l_seq;
    mate_reference_sequence_id[i] = (int)load_u32(buf + r + 24);
    mate_position[i] = (int)load_u32(buf + r + 28);
    template_length[i] = (int)load_u32(buf + r + 32);

    u32 c = NAME_START + name_len;
    u32 s = c + 4u * n_cigar;
    u32 q = s + (l_seq + 1u) / 2u;
    u32 a = q + l_seq;

    cigar_start[i] = c;
    sequence_start[i] = s;
    quality_start[i] = q;
    aux_start[i] = a;
    record_end[i] = 4u + block_size;
}
