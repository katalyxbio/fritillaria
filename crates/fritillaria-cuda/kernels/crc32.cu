// Per-block CRC32, the first vertical slice of the GPU pipeline.
//
// One thread per BGZF block. That is the natural mapping for verification:
// blocks are independent, a real file has thousands of them, and the work per
// block is bounded at 64 KiB. It is deliberately *not* one thread block per
// BGZF block — parallelising a single CRC requires carry-less matrix
// combination, which is worth doing only once this is shown to be a bottleneck.
//
// CRC-32/ISO-HDLC (reflected, polynomial 0xEDB88320), the variant gzip uses.
// Computed bitwise rather than from a table so the kernel stays self-contained
// for NVRTC; swap in a shared-memory table if profiling says to.

extern "C" __global__ void crc32_blocks(
    const unsigned char *data,     // concatenated inflated payloads
    const unsigned long long *offsets, // per-block start offsets into `data`
    const unsigned int *lengths,   // per-block payload lengths
    unsigned int *out,             // per-block CRC32 output
    int num_blocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) {
        return;
    }

    const unsigned char *p = data + offsets[i];
    unsigned int len = lengths[i];

    unsigned int crc = 0xFFFFFFFFu;
    for (unsigned int b = 0; b < len; ++b) {
        crc ^= (unsigned int)p[b];
        for (int k = 0; k < 8; ++k) {
            // Branchless: subtracting the low bit yields an all-ones mask.
            crc = (crc >> 1) ^ (0xEDB88320u & (unsigned int)(-(int)(crc & 1u)));
        }
    }

    out[i] = ~crc;
}
