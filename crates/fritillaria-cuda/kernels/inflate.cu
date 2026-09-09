// DEFLATE inflate, one thread per BGZF block.
//
// # Why one thread per block
//
// A single DEFLATE stream is a serial bitstream: symbol boundaries are only
// known after decoding the previous symbol, and back-references depend on
// already-produced output. There is no cheap intra-block parallelism.
//
// BGZF hands us the parallelism instead: a real BAM has thousands of
// independent members, so one thread per member saturates the device without
// any of the difficulty. This mirrors crc32.cu deliberately.
//
// The cost is that the Huffman tables (~1.3 KiB) are per-thread locals, which
// live in local (i.e. global) memory. The optimisation path, when profiling
// justifies it, is one warp per BGZF block with tables in shared memory and
// the back-reference copies spread across lanes. Correctness first: this must
// be byte-identical to the CPU reference before it is fast.
//
// The decoder follows the structure of Mark Adler's `puff`, which is the
// clearest correct statement of DEFLATE (RFC 1951) and is easy to check
// against. CRC32 is folded in here so verification costs no extra pass.

#define FR_MAXBITS 15    // longest Huffman code
#define FR_MAXLCODES 286 // literal/length codes
#define FR_MAXDCODES 30  // distance codes
#define FR_FIXLCODES 288 // literal/length codes in the fixed table
#define FR_MAXCODES (FR_MAXLCODES + FR_MAXDCODES)

// Status codes, mirrored by InflateStatus on the Rust side. Keep in sync.
#define FR_OK 0
#define FR_ERR_TRUNCATED 1
#define FR_ERR_BLOCK_TYPE 2
#define FR_ERR_STORED_LEN 3
#define FR_ERR_OUTPUT_OVERFLOW 4
#define FR_ERR_BAD_CODE 5
#define FR_ERR_BAD_DISTANCE 6
#define FR_ERR_BAD_TABLE 7
#define FR_ERR_BAD_LENGTH_REPEAT 8
#define FR_ERR_BAD_COUNTS 9

// Length codes 257..285: base length and extra bits.
__device__ __constant__ static const short FR_LEN_BASE[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
__device__ __constant__ static const short FR_LEN_EXTRA[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

// Distance codes 0..29: base distance and extra bits.
__device__ __constant__ static const int FR_DIST_BASE[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289,
    16385, 24577};
__device__ __constant__ static const short FR_DIST_EXTRA[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

// The permuted order in which code-length code lengths appear.
__device__ __constant__ static const short FR_CLEN_ORDER[19] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

struct FrState {
    const unsigned char *in;
    unsigned int in_len;
    unsigned int in_pos;

    unsigned long long bitbuf;
    int bitcnt;

    unsigned char *out;
    unsigned int out_cap;
    unsigned int out_pos;

    int error;
};

// Reads `need` bits, least-significant bit first.
__device__ static int fr_bits(FrState *s, int need)
{
    unsigned long long val = s->bitbuf;
    while (s->bitcnt < need) {
        if (s->in_pos >= s->in_len) {
            s->error = FR_ERR_TRUNCATED;
            return 0;
        }
        val |= (unsigned long long)s->in[s->in_pos++] << s->bitcnt;
        s->bitcnt += 8;
    }
    s->bitbuf = val >> need;
    s->bitcnt -= need;
    return (int)(val & ((1ULL << need) - 1));
}

// Decodes one symbol using a canonical Huffman code.
//
// Walks code lengths shortest-first, comparing the accumulated code against
// the range of codes of that length. No decode table is built, which keeps
// per-thread memory small at the cost of a loop per symbol.
__device__ static int fr_decode(FrState *s, const short *counts, const short *symbols)
{
    int code = 0;
    int first = 0;
    int index = 0;

    for (int len = 1; len <= FR_MAXBITS; ++len) {
        code |= fr_bits(s, 1);
        if (s->error != FR_OK) {
            return -1;
        }
        int count = counts[len];
        if (code - first < count) {
            return symbols[index + (code - first)];
        }
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    return -1;
}

// Builds counts/symbols from a list of code lengths.
//
// Returns 0 for a complete code, >0 for an incomplete one, <0 if the code is
// over-subscribed. Incomplete codes are legal only in one narrow case, which
// the caller checks.
__device__ static int fr_construct(short *counts, short *symbols,
                                   const short *lengths, int n)
{
    for (int len = 0; len <= FR_MAXBITS; ++len) {
        counts[len] = 0;
    }
    for (int symbol = 0; symbol < n; ++symbol) {
        counts[lengths[symbol]]++;
    }
    if (counts[0] == n) {
        return 0; // no codes at all: complete but empty
    }

    // Check against over-subscription: at each length, how many codes remain.
    int left = 1;
    for (int len = 1; len <= FR_MAXBITS; ++len) {
        left <<= 1;
        left -= counts[len];
        if (left < 0) {
            return left;
        }
    }

    short offs[FR_MAXBITS + 1];
    offs[1] = 0;
    for (int len = 1; len < FR_MAXBITS; ++len) {
        offs[len + 1] = offs[len] + counts[len];
    }
    for (int symbol = 0; symbol < n; ++symbol) {
        if (lengths[symbol] != 0) {
            symbols[offs[lengths[symbol]]++] = (short)symbol;
        }
    }

    return left;
}

// Decodes literal/length and distance symbols until end-of-block.
__device__ static void fr_codes(FrState *s,
                                const short *lencnt, const short *lensym,
                                const short *distcnt, const short *distsym)
{
    for (;;) {
        int symbol = fr_decode(s, lencnt, lensym);
        if (s->error != FR_OK) {
            return;
        }
        if (symbol < 0) {
            s->error = FR_ERR_BAD_CODE;
            return;
        }

        if (symbol < 256) {
            if (s->out_pos >= s->out_cap) {
                s->error = FR_ERR_OUTPUT_OVERFLOW;
                return;
            }
            s->out[s->out_pos++] = (unsigned char)symbol;
            continue;
        }

        if (symbol == 256) {
            return; // end of block
        }

        // Length/distance pair.
        symbol -= 257;
        if (symbol >= 29) {
            s->error = FR_ERR_BAD_CODE;
            return;
        }
        int len = FR_LEN_BASE[symbol] + fr_bits(s, FR_LEN_EXTRA[symbol]);
        if (s->error != FR_OK) {
            return;
        }

        symbol = fr_decode(s, distcnt, distsym);
        if (s->error != FR_OK) {
            return;
        }
        if (symbol < 0 || symbol >= FR_MAXDCODES) {
            s->error = FR_ERR_BAD_DISTANCE;
            return;
        }
        int dist = FR_DIST_BASE[symbol] + fr_bits(s, FR_DIST_EXTRA[symbol]);
        if (s->error != FR_OK) {
            return;
        }

        // BGZF members are independent, so the window never reaches back past
        // the start of this block's own output.
        if ((unsigned int)dist > s->out_pos) {
            s->error = FR_ERR_BAD_DISTANCE;
            return;
        }
        if (s->out_pos + (unsigned int)len > s->out_cap) {
            s->error = FR_ERR_OUTPUT_OVERFLOW;
            return;
        }

        // Byte-at-a-time on purpose: overlapping copies (dist < len) are legal
        // and are how DEFLATE encodes runs, so this must not be vectorised
        // into a block move.
        for (int i = 0; i < len; ++i) {
            s->out[s->out_pos] = s->out[s->out_pos - dist];
            s->out_pos++;
        }
    }
}

// Type 0: stored, i.e. literal bytes with no compression.
__device__ static void fr_stored(FrState *s)
{
    s->bitbuf = 0;
    s->bitcnt = 0;

    if (s->in_pos + 4 > s->in_len) {
        s->error = FR_ERR_TRUNCATED;
        return;
    }
    unsigned int len = (unsigned int)s->in[s->in_pos] |
                       ((unsigned int)s->in[s->in_pos + 1] << 8);
    unsigned int nlen = (unsigned int)s->in[s->in_pos + 2] |
                        ((unsigned int)s->in[s->in_pos + 3] << 8);
    s->in_pos += 4;

    // NLEN is the one's complement of LEN; disagreement means corruption.
    if (len != (~nlen & 0xffff)) {
        s->error = FR_ERR_STORED_LEN;
        return;
    }
    if (s->in_pos + len > s->in_len) {
        s->error = FR_ERR_TRUNCATED;
        return;
    }
    if (s->out_pos + len > s->out_cap) {
        s->error = FR_ERR_OUTPUT_OVERFLOW;
        return;
    }

    for (unsigned int i = 0; i < len; ++i) {
        s->out[s->out_pos++] = s->in[s->in_pos++];
    }
}

// Type 1: fixed Huffman codes, defined by the spec rather than the stream.
__device__ static void fr_fixed(FrState *s, short *lencnt, short *lensym,
                                short *distcnt, short *distsym, short *lengths)
{
    for (int symbol = 0; symbol < 144; ++symbol) lengths[symbol] = 8;
    for (int symbol = 144; symbol < 256; ++symbol) lengths[symbol] = 9;
    for (int symbol = 256; symbol < 280; ++symbol) lengths[symbol] = 7;
    for (int symbol = 280; symbol < FR_FIXLCODES; ++symbol) lengths[symbol] = 8;
    fr_construct(lencnt, lensym, lengths, FR_FIXLCODES);

    // 32 five-bit codes, which makes the code complete. Symbols 30 and 31 are
    // invalid and are rejected at use time in fr_codes.
    for (int symbol = 0; symbol < 32; ++symbol) lengths[symbol] = 5;
    fr_construct(distcnt, distsym, lengths, 32);

    fr_codes(s, lencnt, lensym, distcnt, distsym);
}

// Type 2: dynamic Huffman codes, described by a header in the stream.
__device__ static void fr_dynamic(FrState *s, short *lencnt, short *lensym,
                                  short *distcnt, short *distsym, short *lengths)
{
    int nlen = fr_bits(s, 5) + 257;
    int ndist = fr_bits(s, 5) + 1;
    int ncode = fr_bits(s, 4) + 4;
    if (s->error != FR_OK) {
        return;
    }
    if (nlen > FR_MAXLCODES || ndist > FR_MAXDCODES) {
        s->error = FR_ERR_BAD_COUNTS;
        return;
    }

    // The code-length code, itself Huffman-coded, in a permuted order.
    for (int i = 0; i < 19; ++i) {
        lengths[FR_CLEN_ORDER[i]] = 0;
    }
    for (int i = 0; i < ncode; ++i) {
        lengths[FR_CLEN_ORDER[i]] = (short)fr_bits(s, 3);
        if (s->error != FR_OK) {
            return;
        }
    }
    if (fr_construct(lencnt, lensym, lengths, 19) != 0) {
        s->error = FR_ERR_BAD_TABLE; // must be complete
        return;
    }

    // Read literal/length and distance code lengths, with run-length repeats.
    int index = 0;
    while (index < nlen + ndist) {
        int symbol = fr_decode(s, lencnt, lensym);
        if (s->error != FR_OK) {
            return;
        }
        if (symbol < 0) {
            s->error = FR_ERR_BAD_CODE;
            return;
        }

        if (symbol < 16) {
            lengths[index++] = (short)symbol;
            continue;
        }

        short len = 0;
        if (symbol == 16) {
            if (index == 0) {
                s->error = FR_ERR_BAD_LENGTH_REPEAT;
                return;
            }
            len = lengths[index - 1];
            symbol = 3 + fr_bits(s, 2);
        } else if (symbol == 17) {
            symbol = 3 + fr_bits(s, 3);
        } else {
            symbol = 11 + fr_bits(s, 7);
        }
        if (s->error != FR_OK) {
            return;
        }
        if (index + symbol > nlen + ndist) {
            s->error = FR_ERR_BAD_LENGTH_REPEAT;
            return;
        }
        while (symbol-- > 0) {
            lengths[index++] = len;
        }
    }

    // A missing end-of-block code would let a stream run away.
    if (lengths[256] == 0) {
        s->error = FR_ERR_BAD_TABLE;
        return;
    }

    // Incomplete codes are legal only when a single symbol is used.
    int err = fr_construct(lencnt, lensym, lengths, nlen);
    if (err != 0 && (err < 0 || nlen != lencnt[0] + 1)) {
        s->error = FR_ERR_BAD_TABLE;
        return;
    }
    err = fr_construct(distcnt, distsym, lengths + nlen, ndist);
    if (err != 0 && (err < 0 || ndist != distcnt[0] + 1)) {
        s->error = FR_ERR_BAD_TABLE;
        return;
    }

    fr_codes(s, lencnt, lensym, distcnt, distsym);
}

// CRC-32/ISO-HDLC over the produced output.
//
// Duplicated from crc32.cu rather than shared: NVRTC compiles each kernel
// source independently, and a six-line loop is cheaper to repeat than to
// plumb a common header through. If a third copy appears, add the header.
__device__ static unsigned int fr_crc32(const unsigned char *p, unsigned int len)
{
    unsigned int crc = 0xFFFFFFFFu;
    for (unsigned int b = 0; b < len; ++b) {
        crc ^= (unsigned int)p[b];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & (unsigned int)(-(int)(crc & 1u)));
        }
    }
    return ~crc;
}

extern "C" __global__ void inflate_blocks(
    const unsigned char *in_data,        // concatenated deflate payloads
    const unsigned long long *in_offsets,
    const unsigned int *in_lengths,
    unsigned char *out_data,             // concatenated output, prefix-summed
    const unsigned long long *out_offsets,
    const unsigned int *out_caps,        // expected ISIZE per block
    unsigned int *status,                // FR_OK or an FR_ERR_* code
    unsigned int *produced,              // bytes actually written
    unsigned int *crc32_out,             // CRC32 of the produced bytes
    int num_blocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) {
        return;
    }

    // ~1.3 KiB of per-thread tables. Dynamically indexed, so these live in
    // local memory; see the header comment on the optimisation path.
    short lencnt[FR_MAXBITS + 1];
    short lensym[FR_FIXLCODES];
    short distcnt[FR_MAXBITS + 1];
    short distsym[32];
    short lengths[FR_MAXCODES];

    FrState s;
    s.in = in_data + in_offsets[i];
    s.in_len = in_lengths[i];
    s.in_pos = 0;
    s.bitbuf = 0;
    s.bitcnt = 0;
    s.out = out_data + out_offsets[i];
    s.out_cap = out_caps[i];
    s.out_pos = 0;
    s.error = FR_OK;

    // A deflate stream is a sequence of blocks; only the last is flagged.
    int last = 0;
    do {
        last = fr_bits(&s, 1);
        int type = fr_bits(&s, 2);
        if (s.error != FR_OK) {
            break;
        }

        if (type == 0) {
            fr_stored(&s);
        } else if (type == 1) {
            fr_fixed(&s, lencnt, lensym, distcnt, distsym, lengths);
        } else if (type == 2) {
            fr_dynamic(&s, lencnt, lensym, distcnt, distsym, lengths);
        } else {
            s.error = FR_ERR_BLOCK_TYPE; // type 3 is reserved
        }
    } while (s.error == FR_OK && !last);

    status[i] = (unsigned int)s.error;
    produced[i] = s.out_pos;
    crc32_out[i] = (s.error == FR_OK) ? fr_crc32(s.out, s.out_pos) : 0u;
}
