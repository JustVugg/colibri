/* Qwen3.8 tokenizer NFC normalization.  The generated data lives in
 * qwen38_nfc_tables.h; this file owns the small normalization algorithm. */
#ifndef COLI_QWEN38_NFC_H
#define COLI_QWEN38_NFC_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "qwen38_nfc_tables.h"

#define Q38_NFC_RAW_BASE 0x110000u

typedef struct {
    uint32_t *data;
    size_t length, capacity;
} Q38NfcCodepoints;

static int q38_nfc_push(Q38NfcCodepoints *values, uint32_t value) {
    if (values->length == values->capacity) {
        size_t capacity = values->capacity ? values->capacity * 2u : 32u;
        if (capacity < values->capacity || capacity > SIZE_MAX / sizeof(uint32_t))
            return -1;
        uint32_t *grown = (uint32_t *)realloc(values->data,
                                               capacity * sizeof(uint32_t));
        if (!grown) return -1;
        values->data = grown;
        values->capacity = capacity;
    }
    values->data[values->length++] = value;
    return 0;
}

static uint8_t q38_nfc_combining_class(uint32_t codepoint) {
    size_t lo = 0, hi = sizeof(q38_nfc_ccc) / sizeof(q38_nfc_ccc[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2u;
        if (q38_nfc_ccc[mid].cp < codepoint) lo = mid + 1u;
        else hi = mid;
    }
    return lo < sizeof(q38_nfc_ccc) / sizeof(q38_nfc_ccc[0]) &&
           q38_nfc_ccc[lo].cp == codepoint ? q38_nfc_ccc[lo].ccc : 0;
}

static const Q38NfcDecomp *q38_nfc_decomposition(uint32_t codepoint) {
    size_t lo = 0, hi = sizeof(q38_nfc_decomp) / sizeof(q38_nfc_decomp[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2u;
        if (q38_nfc_decomp[mid].cp < codepoint) lo = mid + 1u;
        else hi = mid;
    }
    return lo < sizeof(q38_nfc_decomp) / sizeof(q38_nfc_decomp[0]) &&
           q38_nfc_decomp[lo].cp == codepoint ? &q38_nfc_decomp[lo] : NULL;
}

static int q38_nfc_decompose(Q38NfcCodepoints *out, uint32_t codepoint) {
    enum { SBASE = 0xac00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11a7,
           LCOUNT = 19, VCOUNT = 21, TCOUNT = 28, NCOUNT = VCOUNT * TCOUNT,
           SCOUNT = LCOUNT * NCOUNT };
    if (codepoint >= SBASE && codepoint < SBASE + SCOUNT) {
        uint32_t index = codepoint - SBASE;
        if (q38_nfc_push(out, LBASE + index / NCOUNT) ||
            q38_nfc_push(out, VBASE + (index % NCOUNT) / TCOUNT)) return -1;
        return index % TCOUNT ? q38_nfc_push(out, TBASE + index % TCOUNT) : 0;
    }
    const Q38NfcDecomp *mapping = q38_nfc_decomposition(codepoint);
    if (!mapping) return q38_nfc_push(out, codepoint);
    for (uint8_t index = 0; index < mapping->length; index++)
        if (q38_nfc_decompose(out,
                q38_nfc_decomp_values[mapping->offset + index])) return -1;
    return 0;
}

static uint32_t q38_nfc_composite(uint32_t first, uint32_t second) {
    enum { SBASE = 0xac00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11a7,
           LCOUNT = 19, VCOUNT = 21, TCOUNT = 28, NCOUNT = VCOUNT * TCOUNT,
           SCOUNT = LCOUNT * NCOUNT };
    if (first >= LBASE && first < LBASE + LCOUNT &&
        second >= VBASE && second < VBASE + VCOUNT)
        return SBASE + ((first - LBASE) * VCOUNT + second - VBASE) * TCOUNT;
    if (first >= SBASE && first < SBASE + SCOUNT &&
        (first - SBASE) % TCOUNT == 0 &&
        second > TBASE && second < TBASE + TCOUNT)
        return first + second - TBASE;

    size_t lo = 0, hi = sizeof(q38_nfc_compose) / sizeof(q38_nfc_compose[0]);
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2u;
        const Q38NfcCompose *row = &q38_nfc_compose[mid];
        if (row->first < first || (row->first == first && row->second < second))
            lo = mid + 1u;
        else hi = mid;
    }
    if (lo < sizeof(q38_nfc_compose) / sizeof(q38_nfc_compose[0]) &&
        q38_nfc_compose[lo].first == first &&
        q38_nfc_compose[lo].second == second)
        return q38_nfc_compose[lo].composite;
    return 0;
}

static void q38_nfc_stable_sort_segment(uint32_t *values,uint32_t *scratch,
                                         size_t begin,size_t end) {
    size_t count=end-begin;
    if(count<2)return;
    uint32_t *source=values+begin,*destination=scratch;
    for(size_t width=1;width<count;){
        for(size_t block=0;block<count;block+=2u*width){
            size_t left=block,middle=block+width,right=block+2u*width;
            if(middle>count)middle=count;if(right>count)right=count;
            size_t a=left,b=middle,out=left;
            while(a<middle&&b<right){
                uint8_t ac=q38_nfc_combining_class(source[a]);
                uint8_t bc=q38_nfc_combining_class(source[b]);
                destination[out++]=ac<=bc?source[a++]:source[b++];
            }
            while(a<middle)destination[out++]=source[a++];
            while(b<right)destination[out++]=source[b++];
        }
        uint32_t *swap=source;source=destination;destination=swap;
        if(width>count/2u)break;width*=2u;
    }
    if(source!=values+begin)memcpy(values+begin,source,count*sizeof(uint32_t));
}

static int q38_nfc_decode_utf8(const unsigned char *input, size_t length,
                               Q38NfcCodepoints *out) {
    for (size_t at = 0; at < length;) {
        unsigned char lead = input[at];
        uint32_t codepoint = 0;
        size_t width = 0;
        if (lead < 0x80) { codepoint = lead; width = 1; }
        else if (lead >= 0xc2 && lead <= 0xdf) { codepoint = lead & 0x1fu; width = 2; }
        else if (lead >= 0xe0 && lead <= 0xef) { codepoint = lead & 0x0fu; width = 3; }
        else if (lead >= 0xf0 && lead <= 0xf4) { codepoint = lead & 0x07u; width = 4; }
        if (!width || width > length - at) {
            if (q38_nfc_push(out, Q38_NFC_RAW_BASE + lead)) return -1;
            at++; continue;
        }
        int valid = 1;
        for (size_t index = 1; index < width; index++) {
            unsigned char byte = input[at + index];
            if ((byte & 0xc0u) != 0x80u) { valid = 0; break; }
            codepoint = (codepoint << 6) | (byte & 0x3fu);
        }
        if (valid && ((width == 3 && ((lead == 0xe0 && input[at + 1] < 0xa0) ||
                                      (lead == 0xed && input[at + 1] >= 0xa0))) ||
                      (width == 4 && ((lead == 0xf0 && input[at + 1] < 0x90) ||
                                      (lead == 0xf4 && input[at + 1] >= 0x90)))))
            valid = 0;
        if (!valid) {
            if (q38_nfc_push(out, Q38_NFC_RAW_BASE + lead)) return -1;
            at++; continue;
        }
        if (q38_nfc_decompose(out, codepoint)) return -1;
        at += width;
    }
    return 0;
}

static int q38_nfc_normalize(const char *input, size_t length,
                             char **output, size_t *output_length) {
    Q38NfcCodepoints decomposed = {0}, composed = {0};
    uint32_t *ordering_scratch = NULL;
    if (!output || !output_length || (length && !input) ||
        q38_nfc_decode_utf8((const unsigned char *)input, length, &decomposed))
        goto fail;

    /* Canonical ordering within each starter-delimited segment. A stable merge
     * sort avoids quadratic behavior on adversarial alternating CCC runs. */
    ordering_scratch=decomposed.length?
        (uint32_t*)malloc(decomposed.length*sizeof(uint32_t)):NULL;
    if(decomposed.length&&!ordering_scratch)goto fail;
    for(size_t segment=0;segment<decomposed.length;){
        size_t marks=segment;
        if(!q38_nfc_combining_class(decomposed.data[marks]))marks++;
        size_t end=marks;
        while(end<decomposed.length&&
              q38_nfc_combining_class(decomposed.data[end]))end++;
        q38_nfc_stable_sort_segment(decomposed.data,ordering_scratch,
                                     marks,end);
        segment=end;
    }
    free(ordering_scratch);ordering_scratch=NULL;

    if (decomposed.length) {
        if (q38_nfc_push(&composed, decomposed.data[0])) goto fail;
        size_t starter_position = 0;
        uint32_t starter = composed.data[0];
        uint8_t previous_class = 0;
        for (size_t index = 1; index < decomposed.length; index++) {
            uint32_t value = decomposed.data[index];
            uint8_t current_class = q38_nfc_combining_class(value);
            uint32_t composite = (starter < Q38_NFC_RAW_BASE &&
                                  (previous_class < current_class || !previous_class))
                               ? q38_nfc_composite(starter, value) : 0;
            if (composite) {
                composed.data[starter_position] = composite;
                starter = composite;
            } else {
                if (q38_nfc_push(&composed, value)) goto fail;
                if (!current_class) {
                    starter_position = composed.length - 1u;
                    starter = value;
                }
                previous_class = current_class;
            }
        }
    }

    if (composed.length > (SIZE_MAX - 1u) / 4u) goto fail;
    size_t capacity = composed.length * 4u + 1u;
    char *encoded = (char *)malloc(capacity);
    if (!encoded) goto fail;
    size_t written = 0;
    for (size_t index = 0; index < composed.length; index++) {
        uint32_t value = composed.data[index];
        if (value >= Q38_NFC_RAW_BASE) encoded[written++] = (char)(value - Q38_NFC_RAW_BASE);
        else if (value < 0x80) encoded[written++] = (char)value;
        else if (value < 0x800) {
            encoded[written++] = (char)(0xc0u | (value >> 6));
            encoded[written++] = (char)(0x80u | (value & 0x3fu));
        } else if (value < 0x10000) {
            encoded[written++] = (char)(0xe0u | (value >> 12));
            encoded[written++] = (char)(0x80u | ((value >> 6) & 0x3fu));
            encoded[written++] = (char)(0x80u | (value & 0x3fu));
        } else {
            encoded[written++] = (char)(0xf0u | (value >> 18));
            encoded[written++] = (char)(0x80u | ((value >> 12) & 0x3fu));
            encoded[written++] = (char)(0x80u | ((value >> 6) & 0x3fu));
            encoded[written++] = (char)(0x80u | (value & 0x3fu));
        }
    }
    encoded[written] = '\0';
    free(decomposed.data); free(composed.data);
    *output = encoded; *output_length = written;
    return 0;

fail:
    free(ordering_scratch);
    free(decomposed.data); free(composed.data);
    return -1;
}

#endif /* COLI_QWEN38_NFC_H */
