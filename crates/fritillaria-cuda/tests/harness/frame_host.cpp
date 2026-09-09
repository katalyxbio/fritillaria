// Runs kernels/bgzf_frame.cu on the CPU, so its logic is testable without a GPU.
//
// This exists because of a lesson this project has now learned five times: a
// remote run on a rented VM should be testing the *device*, not finding a typo.
// The framing kernel is straight-line byte writes with no warp cooperation, no
// shared memory and no atomics, so a shim that defines the CUDA keywords away
// and loops over the thread indices executes the real source faithfully.
//
// What this does NOT prove: that NVRTC accepts it, that the launch
// configuration is right, or anything about nvCOMP. Those still need hardware.
// What it does prove is that the bytes are correct — which is the part that
// would otherwise be diagnosed by staring at a hex dump over a billable link.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// --- the CUDA surface bgzf_frame.cu actually uses --------------------------

struct Dim3 {
    unsigned int x;
};
static Dim3 blockIdx{0};
static Dim3 threadIdx{0};
static Dim3 blockDim{1};

// `extern "C"` survives into C++ unchanged, so only this one needs defining
// away — which is the point: the shim edits nothing about the kernel's logic.
#define __global__

#include "bgzf_frame.cu"

// --- a trivial length-prefixed wire format ---------------------------------

static std::vector<unsigned char> slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf((size_t)n);
    if (n > 0 && fread(buf.data(), 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    fclose(f);
    return buf;
}

struct Cursor {
    const unsigned char *p;
    const unsigned char *end;

    unsigned long long u64()
    {
        unsigned long long v = 0;
        if (p + 8 > end) {
            fprintf(stderr, "input truncated\n");
            exit(2);
        }
        memcpy(&v, p, 8);
        p += 8;
        return v;
    }

    std::vector<unsigned char> bytes(size_t n)
    {
        if (p + n > end) {
            fprintf(stderr, "input truncated\n");
            exit(2);
        }
        std::vector<unsigned char> v(p, p + n);
        p += n;
        return v;
    }
};

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input> <output>\n", argv[0]);
        return 2;
    }

    std::vector<unsigned char> raw_input = slurp(argv[1]);
    Cursor cur{raw_input.data(), raw_input.data() + raw_input.size()};

    size_t count = (size_t)cur.u64();
    size_t out_total = (size_t)cur.u64();
    size_t block_dim = (size_t)cur.u64();

    std::vector<unsigned long long> slot_offsets(count), deflate_sizes(count),
        raw_offsets(count), out_offsets(count);
    std::vector<unsigned int> raw_sizes(count), crcs(count);
    std::vector<unsigned char> stored(count);

    for (size_t i = 0; i < count; i++) slot_offsets[i] = cur.u64();
    for (size_t i = 0; i < count; i++) deflate_sizes[i] = cur.u64();
    for (size_t i = 0; i < count; i++) raw_offsets[i] = cur.u64();
    for (size_t i = 0; i < count; i++) out_offsets[i] = cur.u64();
    for (size_t i = 0; i < count; i++) raw_sizes[i] = (unsigned int)cur.u64();
    for (size_t i = 0; i < count; i++) crcs[i] = (unsigned int)cur.u64();
    for (size_t i = 0; i < count; i++) stored[i] = (unsigned char)cur.u64();

    std::vector<unsigned char> deflate = cur.bytes((size_t)cur.u64());
    std::vector<unsigned char> raw = cur.bytes((size_t)cur.u64());

    // Poisoned rather than zeroed: a byte the kernel forgets to write shows up
    // as a mismatch instead of coincidentally agreeing with a zero.
    std::vector<unsigned char> out(out_total, 0xa5);

    blockDim.x = (unsigned int)block_dim;
    for (size_t b = 0; b < count; b++) {
        blockIdx.x = (unsigned int)b;
        for (size_t t = 0; t < block_dim; t++) {
            threadIdx.x = (unsigned int)t;
            frame_blocks(deflate.data(), slot_offsets.data(), deflate_sizes.data(),
                         raw.data(), raw_offsets.data(), raw_sizes.data(), crcs.data(),
                         stored.data(), out.data(), out_offsets.data(), (int)count);
        }
    }

    FILE *f = fopen(argv[2], "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    if (out_total > 0 && fwrite(out.data(), 1, out_total, f) != out_total) {
        fprintf(stderr, "short write\n");
        return 2;
    }
    fclose(f);
    return 0;
}
