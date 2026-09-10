// FASTQ record boundary discovery and columnar decode.
//
// The CPU reference is `fritillaria_fastq::columnar::{speculative, record}` and
// the differential test is `fritillaria-cuda/tests/fastq_decode.rs`. Read
// `docs/fastq-boundaries.md` for why the validator is shaped the way it is.
//
// # How this differs from bcf_scan.cu, and it is the interesting part
//
// BCF's sieve tests *every byte offset*, because a record can begin anywhere.
// FASTQ records begin only just after a newline, so the candidate set is the
// line starts — roughly one in every 60 bytes for Illumina, one in tens of
// thousands for ONT. The sieve is therefore fused with the newline test: thread
// i asks "is buf[i-1] a newline, and does a record start at i?" and appends only
// if both hold. One pass, no intermediate candidate array.
//
// That matters more than it looks. Materialising line starts first would cost a
// second array the size of the line count and a second pass over the batch, to
// save a `buf[i-1]` load that is already in cache from the neighbouring thread.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

// Scans forward for a newline, returning its offset or `len` if there is none.
//
// Deliberately not a `memchr` equivalent: the walks here are bounded by the
// record, and a record's lines are short except for long-read sequence and
// quality, where the two scans are the dominant cost of validating a candidate.
__device__ __forceinline__ u64 line_end(const u8 *buf, u64 len, u64 pos) {
    while (pos < len && buf[pos] != '\n') {
        ++pos;
    }
    return pos;
}

// The four line boundaries of a record at `pos`, or false if it is not one.
//
// A direct translation of `bounds_at` in the reference. Both checks are load-
// bearing and cover different cases: the length check rejects every decoy in
// real data, and the '+' check rejects the ones where a name line and a '+' line
// happen to be the same length. Deleting either is a silent correctness bug --
// deleting the '+' check passed the entire host suite until a hand-built case
// was added for it.
__device__ __forceinline__ bool bounds_at(const u8 *buf, u64 len, u64 pos, u32 *sequence_start,
                                          u32 *plus_start, u32 *quality_start, u32 *sequence_len,
                                          u32 *record_end) {
    if (pos >= len || buf[pos] != '@') {
        return false;
    }

    u64 definition_end = line_end(buf, len, pos);
    if (definition_end >= len) {
        return false; // no newline: the batch ended inside the definition
    }
    u64 seq_start = definition_end + 1;

    u64 sequence_end = line_end(buf, len, seq_start);
    if (sequence_end >= len) {
        return false;
    }
    u64 pl_start = sequence_end + 1;

    if (pl_start >= len || buf[pl_start] != '+') {
        return false;
    }

    u64 plus_end = line_end(buf, len, pl_start);
    if (plus_end >= len) {
        return false;
    }
    u64 qual_start = plus_end + 1;

    u64 seq_len = sequence_end - seq_start;

    // Measure the quality *line*, then require the two lengths to agree.
    // Computing `qual_start + seq_len` instead lets the span run over the
    // trailing newline on the last record, so a short quality line reads as
    // complete. The reference had exactly that bug.
    u64 quality_end = line_end(buf, len, qual_start);
    if (quality_end - qual_start != seq_len) {
        return false;
    }

    // A record ends past its trailing newline when it has one; the final record
    // of a file may not, which is legal.
    u64 end = (quality_end < len) ? quality_end + 1 : quality_end;

    *sequence_start = (u32)(seq_start - pos);
    *plus_start = (u32)(pl_start - pos);
    *quality_start = (u32)(qual_start - pos);
    *sequence_len = (u32)seq_len;
    *record_end = (u32)(end - pos);
    return true;
}

extern "C" {

// Sieve: one thread per byte, appending record starts.
//
// A candidate is `start` itself or any offset just after a newline. Both the
// newline test and the validation happen here, so nothing intermediate is
// written out.
//
// Offsets land in `out` unordered — an atomic append gives no ordering — and the
// host sorts them. That is a few thousand values per batch against the batch's
// tens of megabytes, and it is the same trade `bcf_scan.cu` makes: the transfer
// this project exists to delete scales with the *data*, this one with the
// *record count*.
__global__ void fastq_sieve(const u8 *buf, u64 len, u64 start, u64 *out, u32 *count, u32 capacity,
                            u32 *overflow, u64 *anchor) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    i += start;
    if (i >= len) {
        return;
    }

    // One thread reports where records can actually begin: `start`, advanced
    // past any newlines. A record whose quality line ended exactly at the
    // previous batch's edge leaves its trailing newline here, so the first real
    // record sits at start+1 -- and the host, which does not have the bytes,
    // would otherwise fail the tiling anchor and drop the whole batch onto the
    // serial walk. Correct, and exactly as slow as the design this replaces.
    if (i == start) {
        u64 a = start;
        while (a < len && buf[a] == '\n') {
            ++a;
        }
        *anchor = a;
    }

    // Candidate line starts only. `start` is a candidate because a batch resumes
    // at a record boundary the previous batch reported.
    bool is_line_start = (i == start) || (buf[i - 1] == '\n');
    if (!is_line_start) {
        return;
    }

    u32 unused_a, unused_b, unused_c, unused_d, unused_e;
    if (!bounds_at(buf, len, i, &unused_a, &unused_b, &unused_c, &unused_d, &unused_e)) {
        return;
    }

    u32 slot = atomicAdd(count, 1u);
    if (slot < capacity) {
        out[slot] = i;
    } else {
        atomicAdd(overflow, 1u);
    }
}

// Decode: one thread per proven record, filling the columns.
//
// Runs only after the host has sorted the survivors and proved they tile, so
// every offset here is known to be a real record start. The walk is repeated
// rather than carried over from the sieve because passing five u32s per
// candidate through the atomic append would cost more bandwidth than redoing a
// bounded scan over bytes that are still in cache.
__global__ void fastq_decode(const u8 *buf, u64 len, const u64 *offsets, u32 n,
                             u32 *sequence_start, u32 *plus_start, u32 *quality_start,
                             u32 *sequence_len, u32 *record_end, u32 *errors) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    u32 seq_start, pl_start, qual_start, seq_len, end;
    if (!bounds_at(buf, len, offsets[i], &seq_start, &pl_start, &qual_start, &seq_len, &end)) {
        // Unreachable if the host proved the tiling, which is why it is counted
        // rather than ignored: reaching it means the two disagree, and shipping
        // columns that are wrong for some rows is worse than failing the batch.
        atomicAdd(errors, 1u);
        return;
    }

    sequence_start[i] = seq_start;
    plus_start[i] = pl_start;
    quality_start[i] = qual_start;
    sequence_len[i] = seq_len;
    record_end[i] = end;
}

// Fallback: a single thread walking the record chain.
//
// Reached when the tiling proof fails. Correct, slow, and on device — copying
// the batch back to walk it on the host would pay exactly the transfer this
// library exists to delete. Nothing in real data has triggered it, so it is
// tested directly rather than waited for.
__global__ void fastq_walk(const u8 *buf, u64 len, u64 start, u64 *out, u32 *count, u32 capacity,
                           u32 *overflow, u64 *tail) {
    u64 pos = start;
    u32 n = 0;

    while (pos < len) {
        // Skip a leading newline: a record whose quality line ended exactly at
        // the previous batch's edge leaves its terminator here.
        if (buf[pos] == '\n') {
            ++pos;
            continue;
        }

        u32 a, b, c, d, end;
        if (!bounds_at(buf, len, pos, &a, &b, &c, &d, &end)) {
            break; // partial trailing record, or a resume at a bad offset
        }
        if (n < capacity) {
            out[n] = pos;
        } else {
            atomicAdd(overflow, 1u);
        }
        ++n;
        pos += end;
    }

    *count = n;
    *tail = pos;
}

} // extern "C"
