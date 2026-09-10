// Wrap compressed chunks as BGZF blocks, gathering them into a dense stream.
//
// This is the pass the read path does not need. nvCOMP's *decompression* output
// alignment is 1, so inflate writes straight into a densely packed buffer; its
// *compression* output alignment is 8, and the compressed size of a chunk is not
// known until the kernel has run, so compression has to write into padded
// worst-case slots — 148,256 bytes each, 2.26x a full BGZF payload. Those slots
// are what this kernel gathers out of.
//
// The pass is affordable only because it runs over the *compressed* side. The
// equivalent on the read path would have touched the inflated buffer, the
// largest thing in the pipeline; here it touches roughly a third of the input.
//
// One thread block per BGZF block, so both the copy and the header writes are
// coalesced. The 18 header bytes and 8 trailer bytes are written by a single
// thread each — they are not worth splitting, and having one thread own each
// gives the `BC` computation exactly one place to be wrong.
//
// The host decides `framing[i]`, it is not recomputed here. That is not caution:
// the host has to size the output buffer *before* this kernel runs, so it has
// already made the choice, and deriving it a second time would be a second
// chance to disagree about where a block starts. See
// `fritillaria_core::compress::choose_framing`.

// Gzip header bytes before the deflate stream, and the CRC32 + ISIZE after it.
#define BGZF_HEADER_SIZE 18
#define BGZF_TRAILER_SIZE 8

// Framing codes, matching `fritillaria_core::compress::Framing`.
#define FRAMING_DEFLATED 0
#define FRAMING_STORED 1
#define FRAMING_EMPTY 2

extern "C" __global__ void frame_blocks(
    const unsigned char *deflate,            // nvCOMP output, in padded slots
    const unsigned long long *slot_offsets,  // each chunk's slot in `deflate`
    const unsigned long long *deflate_sizes, // bytes nvCOMP actually produced
    const unsigned char *raw,                // the uncompressed payloads
    const unsigned long long *raw_offsets,   // each chunk's start in `raw`
    const unsigned int *raw_sizes,           // each chunk's uncompressed length
    const unsigned int *crcs,                // CRC32 of each uncompressed chunk
    const unsigned char *framing,            // FRAMING_* per block, chosen by the host
    unsigned char *out,                      // the dense BGZF stream
    const unsigned long long *out_offsets,   // where each block starts in `out`
    int num_blocks)
{
    int i = blockIdx.x;
    if (i >= num_blocks) {
        return;
    }

    unsigned char *block = out + out_offsets[i];
    unsigned int raw_len = raw_sizes[i];
    unsigned char how = framing[i];

    // Body length decides the BC field, so it has to be known before the
    // header is written. A stored block is its payload plus five bytes of
    // DEFLATE framing; that fixed overhead is the whole reason storing is a
    // usable fallback.
    unsigned long long body_len;
    if (how == FRAMING_STORED) {
        body_len = (unsigned long long)raw_len + 5;
    } else if (how == FRAMING_EMPTY) {
        body_len = 2;
    } else {
        body_len = deflate_sizes[i];
    }
    unsigned long long block_size = BGZF_HEADER_SIZE + body_len + BGZF_TRAILER_SIZE;

    if (threadIdx.x == 0) {
        block[0] = 0x1f;
        block[1] = 0x8b;
        block[2] = 0x08; // deflate
        block[3] = 0x04; // FEXTRA
        block[4] = 0x00; // MTIME: zero, so output is reproducible
        block[5] = 0x00;
        block[6] = 0x00;
        block[7] = 0x00;
        block[8] = 0x00; // XFL
        block[9] = 0xff; // OS = unknown
        block[10] = 0x06; // XLEN = 6: the BC subfield and nothing else
        block[11] = 0x00;
        block[12] = 'B';
        block[13] = 'C';
        block[14] = 0x02; // SLEN = 2
        block[15] = 0x00;
        // BC holds the total block size MINUS ONE. That minus-one exists so a
        // full 65536-byte block fits a uint16, and getting it wrong is the
        // classic BGZF bug.
        unsigned int bc = (unsigned int)(block_size - 1);
        block[16] = (unsigned char)(bc & 0xff);
        block[17] = (unsigned char)((bc >> 8) & 0xff);
    }

    unsigned char *body = block + BGZF_HEADER_SIZE;

    if (how == FRAMING_EMPTY) {
        // The canonical DEFLATE encoding of no bytes: one empty fixed-Huffman
        // block. Not whatever the compressor happened to emit — htslib decides
        // a BGZF file is complete by comparing its last 28 bytes against a
        // fixed EOF marker, and these two are that marker's body.
        if (threadIdx.x == 0) {
            body[0] = 0x03;
            body[1] = 0x00;
        }
    } else if (how == FRAMING_STORED) {
        if (threadIdx.x == 0) {
            // BFINAL = 1, BTYPE = 00. Stored blocks are byte-aligned and this
            // is the first byte, so no bit padding is needed.
            body[0] = 0x01;
            body[1] = (unsigned char)(raw_len & 0xff);
            body[2] = (unsigned char)((raw_len >> 8) & 0xff);
            unsigned int nlen = ~raw_len;
            body[3] = (unsigned char)(nlen & 0xff);
            body[4] = (unsigned char)((nlen >> 8) & 0xff);
        }
        const unsigned char *src = raw + raw_offsets[i];
        for (unsigned int b = threadIdx.x; b < raw_len; b += blockDim.x) {
            body[5 + b] = src[b];
        }
    } else {
        const unsigned char *src = deflate + slot_offsets[i];
        unsigned long long len = deflate_sizes[i];
        for (unsigned long long b = threadIdx.x; b < len; b += blockDim.x) {
            body[b] = src[b];
        }
    }

    if (threadIdx.x == 0) {
        // The gzip trailer checksums the bytes that went *in*, not the
        // compressed stream — hence `crcs` being computed over `raw` in a
        // separate pass before this one.
        unsigned char *trailer = block + BGZF_HEADER_SIZE + body_len;
        unsigned int crc = crcs[i];
        trailer[0] = (unsigned char)(crc & 0xff);
        trailer[1] = (unsigned char)((crc >> 8) & 0xff);
        trailer[2] = (unsigned char)((crc >> 16) & 0xff);
        trailer[3] = (unsigned char)((crc >> 24) & 0xff);
        trailer[4] = (unsigned char)(raw_len & 0xff);
        trailer[5] = (unsigned char)((raw_len >> 8) & 0xff);
        trailer[6] = (unsigned char)((raw_len >> 16) & 0xff);
        trailer[7] = (unsigned char)((raw_len >> 24) & 0xff);
    }
}
