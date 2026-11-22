/* qwen3_asr_audio.h - Whisper-style audio frontend for Qwen3-ASR.
 *
 * MVP: 16kHz mono PCM16 WAV decode + log-mel spectrogram matching
 * WhisperFeatureExtractor defaults used by Qwen3-ASR (n_fft=400, hop=160,
 * 128 mel bins).
 */
#ifndef QWEN3_ASR_AUDIO_H
#define QWEN3_ASR_AUDIO_H

#include <math.h>
#include <stdint.h>
#include "qwen3_asr_mel_fb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef Q3ASR_PI
#define Q3ASR_PI 3.14159265358979323846
#endif

typedef struct {
    float *samples;
    size_t n;
    int sample_rate;
} Q3AsrWav;

typedef struct {
    float *mel;      /* [n_frames, n_mels] row-major */
    int n_frames;
    int n_mels;
} Q3AsrMel;

static inline void q3asr_wav_free(Q3AsrWav *w) {
    if (!w) return;
    free(w->samples);
    w->samples = NULL;
    w->n = 0;
}

static inline void q3asr_mel_free(Q3AsrMel *m) {
    if (!m) return;
    free(m->mel);
    m->mel = NULL;
    m->n_frames = 0;
    m->n_mels = 0;
}

static inline uint16_t q3asr_rd_u16(const unsigned char *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static inline uint32_t q3asr_rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline int16_t q3asr_rd_i16(const unsigned char *p) {
    return (int16_t)q3asr_rd_u16(p);
}

static int q3asr_read_wav_bytes(const unsigned char *buf, size_t len, Q3AsrWav *out) {
    memset(out, 0, sizeof(*out));
    if (len < 44 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0)
        return -1;
    size_t pos = 12;
    int fmt_ok = 0;
    uint16_t audio_fmt = 0, channels = 0, bits = 0;
    uint32_t sr = 0;
    const unsigned char *data = NULL;
    uint32_t data_size = 0;
    while (pos + 8 <= len) {
        const unsigned char *ck = buf + pos;
        uint32_t csz = q3asr_rd_u32(ck + 4);
        pos += 8;
        if (pos + csz > len) break;
        if (memcmp(ck, "fmt ", 4) == 0 && csz >= 16) {
            audio_fmt = q3asr_rd_u16(buf + pos);
            channels = q3asr_rd_u16(buf + pos + 2);
            sr = q3asr_rd_u32(buf + pos + 4);
            bits = q3asr_rd_u16(buf + pos + 14);
            fmt_ok = 1;
        } else if (memcmp(ck, "data", 4) == 0) {
            data = buf + pos;
            data_size = csz;
        }
        pos += csz + (csz & 1u);
    }
    if (!fmt_ok || !data || channels == 0) return -2;
    if (audio_fmt != 1 || bits != 16) return -3; /* PCM16 MVP */
    size_t n_samples_total = data_size / (size_t)(bits / 8);
    size_t n_frames = n_samples_total / channels;
    float *samples = (float *)malloc(n_frames * sizeof(float));
    if (!samples) return -4;
    for (size_t i = 0; i < n_frames; i++) {
        float acc = 0.0f;
        for (uint16_t c = 0; c < channels; c++) {
            size_t idx = (i * channels + c) * 2;
            int16_t v = q3asr_rd_i16(data + idx);
            acc += (float)v / 32768.0f;
        }
        samples[i] = acc / (float)channels;
    }
    out->samples = samples;
    out->n = n_frames;
    out->sample_rate = (int)sr;
    return 0;
}

static int q3asr_read_wav_file(const char *path, Q3AsrWav *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) { fclose(f); return -4; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    fclose(f);
    int rc = q3asr_read_wav_bytes(buf, (size_t)sz, out);
    free(buf);
    return rc;
}

/* Simple linear resampler to 16kHz. */
static int q3asr_resample_16k(const Q3AsrWav *in, Q3AsrWav *out) {
    memset(out, 0, sizeof(*out));
    if (!in || !in->samples || in->n == 0 || in->sample_rate <= 0) return -1;
    if (in->sample_rate == 16000) {
        out->samples = (float *)malloc(in->n * sizeof(float));
        if (!out->samples) return -4;
        memcpy(out->samples, in->samples, in->n * sizeof(float));
        out->n = in->n;
        out->sample_rate = 16000;
        return 0;
    }
    double ratio = 16000.0 / (double)in->sample_rate;
    size_t out_n = (size_t)((double)in->n * ratio);
    if (out_n == 0) return -1;
    float *s = (float *)malloc(out_n * sizeof(float));
    if (!s) return -4;
    for (size_t i = 0; i < out_n; i++) {
        double src = (double)i / ratio;
        size_t i0 = (size_t)src;
        size_t i1 = i0 + 1 < in->n ? i0 + 1 : i0;
        double t = src - (double)i0;
        s[i] = (float)((1.0 - t) * in->samples[i0] + t * in->samples[i1]);
    }
    out->samples = s;
    out->n = out_n;
    out->sample_rate = 16000;
    return 0;
}

/* Hann window, periodic. */
static void q3asr_hann(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f - 0.5f * cosf((float)(2.0 * Q3ASR_PI * i / n));
}

/* Naive DFT for first n_fft/2+1 bins. Enough for MVP correctness tests. */
static void q3asr_rfft_mag2(const float *frame, int n_fft, float *out_mag2) {
    int bins = n_fft / 2 + 1;
    for (int k = 0; k < bins; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < n_fft; n++) {
            double ang = 2.0 * Q3ASR_PI * (double)k * (double)n / (double)n_fft;
            re += frame[n] * cos(ang);
            im -= frame[n] * sin(ang);
        }
        out_mag2[k] = (float)(re * re + im * im);
    }
}

/* Slaney mel scale (librosa/Whisper default). */
static float q3asr_hz_to_mel(float hz) {
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    float mel = (hz - f_min) / f_sp;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (hz >= min_log_hz) mel = min_log_mel + logf(hz / min_log_hz) / logstep;
    return mel;
}
static float q3asr_mel_to_hz(float mel) {
    const float f_min = 0.0f;
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - f_min) / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (mel >= min_log_mel) return min_log_hz * expf(logstep * (mel - min_log_mel));
    return f_min + f_sp * mel;
}

/* Slaney-style mel filterbank [n_mels, n_fft/2+1]. */
static float *q3asr_mel_filterbank(int n_mels, int n_fft, int sr) {
    int bins = n_fft / 2 + 1;
    float *fb = (float *)calloc((size_t)n_mels * bins, sizeof(float));
    if (!fb) return NULL;
    /* Use reference WhisperFeatureExtractor filterbank for default Qwen3-ASR config. */
    if (n_mels == 128 && n_fft == 400 && sr == 16000) {
        memcpy(fb, QWEN3_ASR_MEL_FB_128_201, sizeof(QWEN3_ASR_MEL_FB_128_201));
        return fb;
    }
    float fmin_m = q3asr_hz_to_mel(0.0f);
    float fmax_m = q3asr_hz_to_mel((float)sr / 2.0f);
    float *mpts = (float *)malloc((size_t)(n_mels + 2) * sizeof(float));
    float *hz = (float *)malloc((size_t)(n_mels + 2) * sizeof(float));
    int *bin = (int *)malloc((size_t)(n_mels + 2) * sizeof(int));
    if (!mpts || !hz || !bin) { free(fb); free(mpts); free(hz); free(bin); return NULL; }
    for (int i = 0; i < n_mels + 2; i++) {
        mpts[i] = fmin_m + (fmax_m - fmin_m) * (float)i / (float)(n_mels + 1);
        hz[i] = q3asr_mel_to_hz(mpts[i]);
        bin[i] = (int)floorf((float)(n_fft + 1) * hz[i] / (float)sr);
        if (bin[i] < 0) bin[i] = 0;
        if (bin[i] > bins - 1) bin[i] = bins - 1;
    }
    for (int m = 0; m < n_mels; m++) {
        int l = bin[m], c = bin[m + 1], r = bin[m + 2];
        for (int k = l; k < c; k++) {
            if (c != l) fb[m * bins + k] = ((float)k - (float)l) / (float)(c - l);
        }
        for (int k = c; k < r; k++) {
            if (r != c) fb[m * bins + k] = ((float)r - (float)k) / (float)(r - c);
        }
        /* Slaney area normalization */
        float enorm = 2.0f / (hz[m + 2] - hz[m]);
        for (int k = 0; k < bins; k++) fb[m * bins + k] *= enorm;
    }
    free(mpts); free(hz); free(bin);
    return fb;
}

/* Whisper-style log-mel: pad to 30s? MVP does exact frames from input. */
static int q3asr_log_mel(const float *samples, size_t n_samples, int sr,
                         int n_fft, int hop, int n_mels, Q3AsrMel *out) {
    memset(out, 0, sizeof(*out));
    if (sr != 16000) return -1;
    if (n_samples == 0) return -1;
    int bins = n_fft / 2 + 1;
    int n_frames = (int)(n_samples / (size_t)hop); if (n_frames < 1) n_frames = 1;
    float *win = (float *)malloc((size_t)n_fft * sizeof(float));
    float *frame = (float *)malloc((size_t)n_fft * sizeof(float));
    float *mag2 = (float *)malloc((size_t)bins * sizeof(float));
    float *fb = q3asr_mel_filterbank(n_mels, n_fft, sr);
    float *mel = (float *)malloc((size_t)n_frames * n_mels * sizeof(float));
    if (!win || !frame || !mag2 || !fb || !mel) {
        free(win); free(frame); free(mag2); free(fb); free(mel); return -4;
    }
    q3asr_hann(win, n_fft);
    for (int t = 0; t < n_frames; t++) {
        int start = t * hop - n_fft / 2;
        for (int i = 0; i < n_fft; i++) {
            int idx = start + i;
            /* reflect padding (Whisper center=True, pad_mode=reflect) */
            while (idx < 0 || (size_t)idx >= n_samples) {
                if (idx < 0) idx = -idx;
                else idx = (int)(2 * n_samples - (size_t)idx - 2);
                if (n_samples <= 1) { idx = 0; break; }
            }
            frame[i] = samples[idx] * win[i];
        }
        q3asr_rfft_mag2(frame, n_fft, mag2);
        for (int m = 0; m < n_mels; m++) {
            double acc = 0.0;
            const float *frow = fb + (size_t)m * bins;
            for (int k = 0; k < bins; k++) acc += (double)frow[k] * (double)mag2[k];
            float v = (float)acc;
            if (v < 1e-10f) v = 1e-10f;
            mel[(size_t)t * n_mels + m] = log10f(v);
        }
    }
    /* Whisper normalization: max - 8, clamp to -8 (rough MVP; refine to exact). */
    float mx = mel[0];
    for (size_t i = 1; i < (size_t)n_frames * n_mels; i++) if (mel[i] > mx) mx = mel[i];
    float floorv = mx - 8.0f;
    for (size_t i = 0; i < (size_t)n_frames * n_mels; i++) {
        if (mel[i] < floorv) mel[i] = floorv;
        mel[i] = (mel[i] + 4.0f) / 4.0f;
    }
    out->mel = mel;
    out->n_frames = n_frames;
    out->n_mels = n_mels;
    free(win); free(frame); free(mag2); free(fb);
    return 0;
}

#endif /* QWEN3_ASR_AUDIO_H */