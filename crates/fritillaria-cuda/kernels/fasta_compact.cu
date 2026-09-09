// FASTA compaction: a reference genome made contiguous in device memory.
//
// The CPU reference is `fritillaria_fasta::columnar` and the differential test
// is `fritillaria-cuda/tests/fasta_compact.rs`.
//
// # This is not like the other three kernels
//
// bam_decode, bcf_scan and fastq_scan all *find* things and emit offsets,
// leaving the payload where it is. This one moves the payload, and that is the
// whole point: FASTA wraps its sequence across lines, so a consumer reading 150
// bases hits a newline every 70 and cannot issue a coalesced load. Compacting
// costs a copy of 98.6% of the file and buys contiguity — plus it is the
// precondition for 2-bit packing, which is what makes a 3.1 Gbp genome sit in
// 775 MB rather than 3.1 GB.
//
// # Why no prefix sum
//
// The obvious implementation of "remove every newline" is a stream compaction:
// a flag array, an exclusive scan, then a scatter. That is three passes and a
// cooperative scan.
//
// None of it is needed when the wrapping is uniform. For a source index i
// measured from the start of a contig's sequence span, with
// line_width = line_bases + 1, the number of newlines before i is exactly
// i / line_width, so the destination is i - i / line_width. One thread per
// byte, O(1) each, no cooperation and no scratch.
//
// Uniform wrapping is exactly the condition `samtools faidx` requires before it
// will index a FASTA, so this costs nothing htslib does not also refuse. A
// non-uniform contig is detected on the host and takes `fasta_compact_scan`
// below instead.
//
// # The newline test is a load, not arithmetic
//
// `i % line_width == line_bases` identifies every *interior* newline and misses
// the last one of each contig, because the final line is short so its
// terminator sits at an irregular offset. That is every multi-line contig, not
// an edge case. The byte has to be loaded to be copied anyway, so testing it
// costs nothing.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

extern "C" {

// One thread per source byte of one contig's sequence span.
//
// `src` points at the contig's first sequence byte, `dst` at where its bases
// belong in the compacted reference. Threads that land on a newline simply
// return, so the writes are dense with a hole only where a newline was — and
// since destinations are computed rather than allocated, there is no hole.
__global__ void fasta_compact_uniform(const u8 *src, u64 span, u8 *dst, u32 line_width) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= span) {
        return;
    }
    u8 byte = src[i];
    if (byte == '\n') {
        return;
    }
    dst[i - i / (u64)line_width] = byte;
}

// The fallback, for a contig whose lines are not all the same length.
//
// Single-threaded, because without uniform wrapping a byte's destination
// depends on how many newlines precede it and that is a running count. Rare by
// construction: `samtools faidx` refuses to index such a file, so anything
// reaching here is already outside what htslib will handle.
//
// Kept on device rather than falling back to the host for the same reason every
// other fallback in this workspace is: copying the batch back would pay exactly
// the transfer this library exists to delete.
__global__ void fasta_compact_scan(const u8 *src, u64 span, u8 *dst) {
    u64 at = 0;
    for (u64 i = 0; i < span; ++i) {
        u8 byte = src[i];
        if (byte != '\n') {
            dst[at++] = byte;
        }
    }
}

// Locates contig starts: '>' at the beginning of a line.
//
// Unambiguous, unlike FASTQ's '@' — a sequence line never begins with '>', so
// there is no decoy case, no second check, and no tiling proof. One thread per
// byte, atomic append, and the host sorts the handful of results.
//
// A reference has thousands of contigs at most, against billions of bases, so
// this list is tiny and its round trip to the host is not the transfer this
// project cares about.
__global__ void fasta_find_contigs(const u8 *buf, u64 len, u64 start, u64 *out, u32 *count,
                                   u32 capacity, u32 *overflow) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    i += start;
    if (i >= len || buf[i] != '>') {
        return;
    }
    if (i != start && buf[i - 1] != '\n') {
        return; // a '>' inside a definition line is not a record start
    }

    u32 slot = atomicAdd(count, 1u);
    if (slot < capacity) {
        out[slot] = i;
    } else {
        atomicAdd(overflow, 1u);
    }
}

} // extern "C"
