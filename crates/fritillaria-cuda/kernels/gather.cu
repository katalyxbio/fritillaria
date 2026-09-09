// Copy each block's DEFLATE payload into an aligned slot.
//
// nvCOMP requires every input chunk pointer to be 4-byte aligned (queried at
// runtime; 4 is what nvCOMP 5.3 reports for deflate). A BGZF payload starts 18
// bytes into its gzip member — 10 bytes of fixed header, 2 of XLEN, 6 of `BC`
// subfield — and members sit at arbitrary file offsets, so in general nothing
// guarantees that alignment. This kernel restages the payloads at offsets that
// do, and is skipped entirely when the batch already satisfies the requirement.
//
// The alternative was to compact on the host before upload, which costs a
// memcpy of the whole compressed batch across host memory *and* buys nothing;
// this runs at device memory bandwidth over data already in VRAM.
//
// One thread block per BGZF block, so the copy within a block is coalesced.
// That is a different mapping from inflate.cu (one *thread* per BGZF block),
// and deliberately: inflate is inherently serial per block, a memcpy is not.

extern "C" __global__ void gather_payloads(
    const unsigned char *src,           // the uploaded compressed batch
    const unsigned long long *src_offsets, // payload start of each block in `src`
    const unsigned int *lengths,        // payload length of each block
    unsigned char *dst,                 // aligned staging buffer
    const unsigned long long *dst_offsets, // aligned slot of each block in `dst`
    int num_blocks)
{
    int i = blockIdx.x;
    if (i >= num_blocks) {
        return;
    }

    const unsigned char *in = src + src_offsets[i];
    unsigned char *out = dst + dst_offsets[i];
    unsigned int len = lengths[i];

    for (unsigned int b = threadIdx.x; b < len; b += blockDim.x) {
        out[b] = in[b];
    }
}
