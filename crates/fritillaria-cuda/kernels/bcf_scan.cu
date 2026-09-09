// BCF record boundary discovery, on device.
//
// A direct translation of `fritillaria-bcf/src/speculative.rs`, which is the
// CPU reference and the oracle these kernels are diffed against. Read that
// module first; it carries the correctness argument.
//
// The short version, and why this is not `bam_decode.cu`. BAM speculates at
// BGZF block starts because htslib ends a block early rather than splitting an
// alignment, so a block start is almost always a record start. `bcf_write`
// packs blocks full instead: measured on real files, ZERO of 56 interior block
// starts are record starts. Every guess would be wrong and the scan would
// collapse to one serial chain.
//
// So BCF speculates at *every byte offset* and prunes with a validator, then
// proves the result by tiling rather than by walking a chain:
//
//   bcf_sieve      one thread per byte      fixed-cost 32-byte plausibility
//   bcf_validate   one thread per survivor  full typed-value chain walk
//
// If the survivors tile the buffer, they ARE the true record chain — the true
// chain starts at the same known offset and each record's length is determined
// by its own bytes, so its successor is exactly what the tiling asserts. That
// makes the proof O(1) per record, where BAM's reconcile is a serial walk over
// blocks. A failed tiling is not an error: the caller falls back to the serial
// walk, so a wrong guess costs work and never accuracy.
//
// **The tiling proof is deliberately not a kernel.** Both phases here reduce
// the whole batch to a few thousand offsets, and proving a tiling over those is
// microseconds anywhere. Putting it on device would mean keeping the offsets in
// VRAM, which serves nothing until a device-side columnar BCF decode exists to
// consume them — and building plumbing for a consumer that does not exist is
// how untested code ships. It belongs here when that decode does.
//
// Loads are byte-wise throughout. A BCF record starts wherever the previous one
// ended, so there is no alignment guarantee and reinterpreting a pointer as
// `unsigned int*` would be undefined.

#define FR_BCF_OK 0u
#define FR_BCF_MALFORMED 1u

// Fixed core after the two length prefixes: CHROM through n_fmt.
#define SITE_CORE_SIZE 24u
// A record's smallest possible size, both length prefixes included.
#define MIN_RECORD_SIZE 32u

// BCF2 type codes. 4, 6 and 8-15 are reserved; rejecting them is most of what
// stops a walk wandering through arbitrary bytes.
#define T_MISSING 0u
#define T_INT8 1u
#define T_INT16 2u
#define T_INT32 3u
#define T_FLOAT 5u
#define T_CHAR 7u

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

__device__ __forceinline__ u32 load_u16(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8);
}

__device__ __forceinline__ u32 load_u24(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16);
}

__device__ __forceinline__ u32 load_u32(const u8 *p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

// Bytes per element of a BCF2 type, or 0 for MISSING and reserved codes.
__device__ __forceinline__ u32 type_size(u32 code) {
    switch (code) {
    case T_MISSING:
        return 0u;
    case T_INT8:
    case T_CHAR:
        return 1u;
    case T_INT16:
        return 2u;
    case T_INT32:
    case T_FLOAT:
        return 4u;
    default:
        return 0xFFFFFFFFu; // reserved: not a type
    }
}

// Where the record at `pos` ends, or 0 if its length words are unreadable.
//
// Zero is never a valid end because every record is at least 32 bytes, so it
// doubles as the failure value without a second output.
__device__ __forceinline__ u64 record_end(const u8 *buf, u64 len, u64 pos) {
    if (pos + 8ull > len) {
        return 0ull;
    }
    u64 l_shared = (u64)load_u32(buf + pos);
    u64 l_indiv = (u64)load_u32(buf + pos + 4);
    return pos + 8ull + l_shared + l_indiv;
}

// The fixed-cost plausibility test: `looks_like_a_record` in the reference.
//
// Reads the 32-byte prefix and nothing else, with no data-dependent loop, so
// every thread costs the same regardless of what it lands on. That is the whole
// point — this runs at one thread per byte, and a data-dependent version would
// serialise the warp on whichever thread found the longest garbage.
//
// `samples` and `contigs` come from the header. `samples` is the strong check
// and has no BAM analogue: the spec requires every record's n_sample to equal
// the header's, so a random offset must match a specific 3-byte value.
__device__ __forceinline__ bool looks_like_a_record(const u8 *buf, u64 len, u64 pos, u32 samples,
                                                    u32 contigs) {
    if (pos + MIN_RECORD_SIZE > len) {
        return false;
    }

    u32 l_shared = load_u32(buf + pos);
    u32 l_indiv = load_u32(buf + pos + 4);
    if (l_shared < SITE_CORE_SIZE) {
        return false;
    }
    u64 end = pos + 8ull + (u64)l_shared + (u64)l_indiv;
    if (end > len) {
        return false;
    }

    int chrom = (int)load_u32(buf + pos + 8);
    int position = (int)load_u32(buf + pos + 12);
    int rlen = (int)load_u32(buf + pos + 16);
    if (chrom < 0 || (u32)chrom >= contigs) {
        return false;
    }
    // POS is 0-based; -1 marks a telomere record and nothing below is
    // representable.
    if (position < -1 || rlen < 0) {
        return false;
    }

    u32 n_allele = load_u16(buf + pos + 26);
    u32 n_sample = load_u24(buf + pos + 28);
    u32 n_fmt = (u32)buf[pos + 31];
    if (n_sample != samples || n_allele == 0u) {
        return false;
    }
    // With no FORMAT keys there is no genotype block; with some, each needs at
    // least a key byte and a descriptor byte.
    if (n_fmt == 0u) {
        return l_indiv == 0u;
    }
    return (u64)l_indiv >= (u64)n_fmt * 2ull;
}

// Reads a typed-value descriptor at `pos`.
//
// Writes the element count and type through the out-params and returns the
// descriptor's own length (1, or more when the count escapes). Returns 0 on a
// malformed descriptor, which is never a valid length.
__device__ __forceinline__ u32 read_descriptor(const u8 *buf, u64 len, u64 pos, u32 *count,
                                               u32 *kind) {
    if (pos >= len) {
        return 0u;
    }
    u32 descriptor = buf[pos];
    u32 code = descriptor & 0x0Fu;
    if (type_size(code) == 0xFFFFFFFFu) {
        return 0u; // reserved type
    }
    u32 nibble = descriptor >> 4;
    *kind = code;

    if (nibble < 15u) {
        *count = nibble;
        return 1u;
    }

    // The escape: a count of 15 means "15 or more, and the true count is the
    // next typed integer". Taking the nibble at face value would read 15
    // elements of a 200-element field and resynchronise onto garbage.
    if (pos + 1ull >= len) {
        return 0u;
    }
    u32 inner = buf[pos + 1];
    u32 inner_code = inner & 0x0Fu;
    if ((inner >> 4) != 1u || (inner_code != T_INT8 && inner_code != T_INT16 &&
                               inner_code != T_INT32)) {
        return 0u; // the length must be an atomic integer
    }
    u32 width = type_size(inner_code);
    if (pos + 2ull + (u64)width > len) {
        return 0u;
    }
    const u8 *p = buf + pos + 2;
    int value;
    if (inner_code == T_INT8) {
        value = (int)(signed char)p[0];
    } else if (inner_code == T_INT16) {
        value = (int)(short)load_u16(p);
    } else {
        value = (int)load_u32(p);
    }
    // 15 is the smallest count the escape may encode; anything less signals a
    // misparse far more often than a quirky writer.
    if (value < 15) {
        return 0u;
    }
    *count = (u32)value;
    return 2u + width;
}

// Steps over one typed value, returning the offset just past it, or 0.
__device__ __forceinline__ u64 skip_typed(const u8 *buf, u64 len, u64 pos) {
    u32 count = 0u;
    u32 kind = 0u;
    u32 header = read_descriptor(buf, len, pos, &count, &kind);
    if (header == 0u) {
        return 0ull;
    }
    u64 end = pos + (u64)header + (u64)count * (u64)type_size(kind);
    return (end > len) ? 0ull : end;
}

// The full validation walk: `Record::validate` in the reference.
//
// Requires the site fields to end EXACTLY at l_shared and the genotype block
// exactly at l_indiv. A misread descriptor desynchronises the walk, so it
// cannot survive this — which is what makes it a real second stage rather than
// a repeat of the sieve.
//
// Unlike the sieve this is data-dependent and its cost is proportional to the
// record, so it runs only on what the sieve let through.
__device__ __forceinline__ bool validate_record(const u8 *buf, u64 len, u64 pos) {
    if (pos + MIN_RECORD_SIZE > len) {
        return false;
    }
    u64 l_shared = (u64)load_u32(buf + pos);
    u64 l_indiv = (u64)load_u32(buf + pos + 4);
    u64 indiv_start = pos + 8ull + l_shared;
    u64 end = indiv_start + l_indiv;
    if (l_shared < SITE_CORE_SIZE || end > len) {
        return false;
    }

    u32 n_info = load_u16(buf + pos + 24);
    u32 n_allele = load_u16(buf + pos + 26);
    u32 n_sample = load_u24(buf + pos + 28);
    u32 n_fmt = (u32)buf[pos + 31];

    u64 at = pos + 8ull + SITE_CORE_SIZE;
    at = skip_typed(buf, indiv_start, at); // ID
    if (at == 0ull) {
        return false;
    }
    for (u32 i = 0; i < n_allele; ++i) {
        at = skip_typed(buf, indiv_start, at);
        if (at == 0ull) {
            return false;
        }
    }
    at = skip_typed(buf, indiv_start, at); // FILTER
    if (at == 0ull) {
        return false;
    }
    for (u32 i = 0; i < n_info; ++i) {
        at = skip_typed(buf, indiv_start, at); // key
        if (at == 0ull) {
            return false;
        }
        at = skip_typed(buf, indiv_start, at); // value
        if (at == 0ull) {
            return false;
        }
    }
    if (at != indiv_start) {
        return false;
    }

    // FORMAT departs from every other typed value: one descriptor covers every
    // sample's slot, so the payload is n_sample times what the descriptor alone
    // implies.
    for (u32 i = 0; i < n_fmt; ++i) {
        at = skip_typed(buf, end, at); // key
        if (at == 0ull) {
            return false;
        }
        u32 count = 0u;
        u32 kind = 0u;
        u32 header = read_descriptor(buf, end, at, &count, &kind);
        if (header == 0u) {
            return false;
        }
        u64 total = (u64)count * (u64)type_size(kind) * (u64)n_sample;
        at += (u64)header + total;
        if (at > end) {
            return false;
        }
    }
    return at == end;
}

extern "C" {

// Phase 1: one thread per byte offset in [start, len).
//
// Survivors are appended with an atomic rather than written to a flag array,
// which is the difference between returning a few thousand offsets and
// returning a byte per input byte. At ~12 survivors per 64 KiB the contention
// is negligible and the output is small enough to leave the device cheaply.
//
// The order is therefore arbitrary; the host sorts. That costs nothing at this
// size and is what lets the whole compaction be one atomic instead of a
// two-level prefix sum.
//
// `capacity` bounds the output. Overflow is reported rather than clamped: a
// truncated survivor list would fail the tiling proof and silently send the
// batch to the serial walk, which is the failure mode this project keeps
// warning itself about — slow, correct, and indistinguishable from working.
// Each survivor's END is written alongside it. The thread has just read the
// two length words to decide, so emitting the sum costs nothing here and saves
// the host a scattered read per candidate later — which at a few thousand
// candidates would be thousands of tiny transfers, far more expensive than the
// sieve itself.
__global__ void bcf_sieve(const u8 *buf, u64 len, u64 start, u32 samples, u32 contigs, u64 *out,
                          u64 *ends, u32 *count, u32 capacity, u32 *overflow) {
    u64 index = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 pos = start + index;
    if (pos >= len) {
        return;
    }
    if (!looks_like_a_record(buf, len, pos, samples, contigs)) {
        return;
    }
    u32 slot = atomicAdd(count, 1u);
    if (slot < capacity) {
        out[slot] = pos;
        ends[slot] = record_end(buf, len, pos);
    } else {
        atomicAdd(overflow, 1u);
    }
}

// The one lookup the tiling proof needs that no survivor carries: whether a
// whole record begins at `pos`, where `pos` is the end of the last survivor.
//
// If one does, the sieve missed it and adopting the tiling would silently drop
// a record — the single way this design can lose data rather than lose speed.
// Anything else there is a genuine partial record, which is the *common* case
// at a BCF batch edge rather than an exceptional one.
__global__ void bcf_probe_tail(const u8 *buf, u64 len, u64 pos, u32 *whole_record) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    u64 end = record_end(buf, len, pos);
    *whole_record = (end != 0ull && end <= len) ? 1u : 0u;
}

// Phase 2: one thread per survivor, appending those whose typed-value chain
// agrees with their declared lengths.
//
// Measured on four real files totalling 399M candidate offsets, this pruned
// nothing: the sieve was already exact. It is kept because without it a single
// false positive breaks the tiling and sends a whole batch to the serial walk.
// See `docs/bcf-boundaries.md`; a caller that would rather have the extra
// fallbacks can skip this launch.
__global__ void bcf_validate(const u8 *buf, u64 len, const u64 *candidates, const u64 *cand_ends,
                             u32 n, u64 *out, u64 *ends, u32 *count) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    u64 pos = candidates[index];
    if (validate_record(buf, len, pos)) {
        u32 slot = atomicAdd(count, 1u);
        out[slot] = pos;
        ends[slot] = cand_ends[index];
    }
}

// The fallback: one thread following the true record chain.
//
// Reached only when the tiling proof fails, which no real file has yet caused.
// It exists so that a failure costs a slow device pass rather than a
// device-to-host copy of the whole batch — the transfer this library exists to
// delete. `bcf_walk_is_the_fallback_and_agrees` exercises it directly rather
// than waiting for data that triggers it.
//
// This is the serial chain the rest of the file is built to avoid, so it is
// also the honest measure of what the speculation buys.
__global__ void bcf_walk(const u8 *buf, u64 len, u64 start, u64 *out, u32 *count, u32 capacity,
                         u32 *overflow, u64 *tail) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    u64 pos = start;
    u32 n = 0u;
    while (true) {
        if (pos + 8ull > len) {
            break; // fewer than 8 bytes left: a partial prefix
        }
        u64 l_shared = (u64)load_u32(buf + pos);
        u64 l_indiv = (u64)load_u32(buf + pos + 4);
        if (l_shared < SITE_CORE_SIZE) {
            break; // malformed; the host reports it after seeing the tail
        }
        u64 end = pos + 8ull + l_shared + l_indiv;
        if (end > len) {
            break; // a partial trailing record: the common case at a batch edge
        }
        if (n < capacity) {
            out[n] = pos;
        } else {
            atomicAdd(overflow, 1u);
        }
        ++n;
        pos = end;
    }
    *count = n;
    *tail = pos;
}

} // extern "C"
