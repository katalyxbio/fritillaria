// Line and field discovery for tab-delimited genomic text.
//
// The CPU reference is `fritillaria_text::columnar` and the differential test
// is `fritillaria-cuda/tests/text_scan.rs`. One kernel set serves SAM, VCF,
// BED, GFF and GTF, because their record framing is identical and only the
// interpretation of the fields differs.
//
// # Three phases, and why not one
//
//   1. fields are found per *line*, and
//   2. a line's fields need somewhere to go, and
//   3. how much room each line needs is only known after counting.
//
// So: find lines, count each line's tabs, prefix-sum the counts into a flat
// table, then write the tabs. Phases 1, 2 and 4 are all embarrassingly
// parallel; phase 3 is a scan over the *line* count, which the host does
// alongside the sort it already has to do.
//
// The alternative -- append every tab atomically and sort -- was rejected on
// scale. Lines scale with records, which this workspace already round-trips.
// Tabs scale with fields: ten times more on SAM, and on a 2504-sample VCF more
// than two thousand times more. A fixed stride per line is wrong for the same
// reason: the table would be sized for the widest record and mostly empty.
//
// # No quote handling, deliberately
//
// A field cannot contain a newline, because a record is a line. A field cannot
// contain a tab, because all five specs forbid it. And `"` is *not* a quote
// character in SAM, VCF, BED or GFF3 -- in SAM it is Phred+33 Q1 and fills the
// quality strings. Only GTF quotes, inside its final field, after every tab;
// measured on a real NCBI-written GTF, zero of 357 records have a tab inside a
// quoted value. So the scan is two byte tests and no state machine.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

// A line is a record unless its first byte marks it as a header.
__device__ __forceinline__ bool is_header(u8 first, u32 comment, u32 secondary) {
    return (u32)first == comment || ((u32)first == secondary && secondary < 256u);
}

extern "C" {

// Phase 1: one thread per byte, appending line starts.
//
// A line begins at `start`, or just after a newline. Blank lines are skipped
// here rather than filtered later -- they carry neither a record nor a header,
// and every one of these formats tolerates them.
//
// `kinds[i]` is 1 for a header line and 0 for a record, keyed to the *unsorted*
// slot, so the host must permute it alongside the offsets when it sorts.
__global__ void text_find_lines(const u8 *buf, u64 len, u64 start, u32 comment, u32 secondary,
                                u64 *out, u8 *kinds, u32 *count, u32 capacity, u32 *overflow) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    i += start;
    if (i >= len) {
        return;
    }

    bool line_start = (i == start) || (buf[i - 1] == '\n');
    if (!line_start || buf[i] == '\n') {
        return; // not a line start, or a blank line
    }

    u32 slot = atomicAdd(count, 1u);
    if (slot < capacity) {
        out[slot] = i;
        kinds[slot] = is_header(buf[i], comment, secondary) ? 1u : 0u;
    } else {
        atomicAdd(overflow, 1u);
    }
}

// Phase 2: one thread per record, counting its tabs and finding its end.
//
// The end excludes the newline, so a caller slicing `[offset, end)` gets the
// record without its terminator. A record with no newline runs to `len`, which
// is the batch-edge case the host resolves by carrying it forward.
__global__ void text_count_fields(const u8 *buf, u64 len, const u64 *offsets, u32 n, u32 *counts,
                                  u64 *ends) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) {
        return;
    }

    u64 at = offsets[r];
    u32 tabs = 0;
    while (at < len && buf[at] != '\n') {
        if (buf[at] == '\t') {
            ++tabs;
        }
        ++at;
    }
    counts[r] = tabs;
    ends[r] = at;
}

// Phase 4: one thread per record, writing its tabs into the slice the prefix
// sum reserved for it.
//
// `starts` has n + 1 entries, so `starts[r]..starts[r + 1]` is record r's span
// and the last entry is the total. Writing is dense and each thread owns its
// span exclusively, so no atomics.
__global__ void text_write_fields(const u8 *buf, const u64 *offsets, const u64 *ends, u32 n,
                                  const u64 *starts, u64 *tabs) {
    u32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) {
        return;
    }

    u64 slot = starts[r];
    u64 limit = starts[r + 1];
    for (u64 at = offsets[r]; at < ends[r]; ++at) {
        if (buf[at] == '\t') {
            if (slot >= limit) {
                return; // counted and written disagree; the host checks totals
            }
            tabs[slot++] = at;
        }
    }
}

} // extern "C"
