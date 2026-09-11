#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

struct capture_state {
    std::string directory;
    bool trace_operations = false;
    bool layer0 = false;
    bool layer8 = false;
    bool layer17 = false;
    bool layer26 = false;
    bool projected = false;
    size_t projected_width = 0;
    size_t projected_tokens = 0;
    size_t operation_count = 0;
};

struct operation_capture {
    const char *tensor;
    const char *file;
};

static constexpr operation_capture operation_captures[] = {
    { "inp_raw_scaled",         "llama-vision-scaled-input.f32" },
    { "pos_embd",              "llama-vision-input.f32" },
    { "layer_inp_normed-0",  "llama-vision-layer0-attention-input-norm.f32" },
    { "Qcur-0",              "llama-vision-layer0-query-norm.f32" },
    { "Kcur-0",              "llama-vision-layer0-key-norm.f32" },
    { "Vcur-0",              "llama-vision-layer0-value.f32" },
    { "Qcur_pos-0",          "llama-vision-layer0-query-rope.f32" },
    { "Kcur_pos-0",          "llama-vision-layer0-key-rope.f32" },
    { "Vcur_normed-0",       "llama-vision-layer0-value-norm.f32" },
    { "kqv_out-0",           "llama-vision-layer0-attention-context.f32" },
    { "attn_out-0",          "llama-vision-layer0-attention-output.f32" },
    { "attn_post_normed-0",  "llama-vision-layer0-attention-post-norm.f32" },
    { "ffn_inp-0",           "llama-vision-layer0-attention-residual.f32" },
    { "ffn_inp_normed-0",    "llama-vision-layer0-ffn-input-norm.f32" },
    { "ffn_up-0",            "llama-vision-layer0-ffn-up.f32" },
    { "ffn_gate-0",          "llama-vision-layer0-ffn-gate.f32" },
    { "ffn_geglu_quick-0",   "llama-vision-layer0-ffn-activated.f32" },
    { "ffn_out-0",           "llama-vision-layer0-ffn-output.f32" },
    { "ffn_post_normed-0",   "llama-vision-layer0-ffn-post-norm.f32" },
    { "layer_out-16",        "llama-vision-layer17.f32" },
    { "layer_inp_normed-17", "llama-vision-layer17-attention-input-norm.f32" },
    { "Qcur-17",             "llama-vision-layer17-query-norm.f32" },
    { "Kcur-17",             "llama-vision-layer17-key-norm.f32" },
    { "Vcur-17",             "llama-vision-layer17-value.f32" },
    { "Qcur_pos-17",         "llama-vision-layer17-query-rope.f32" },
    { "Kcur_pos-17",         "llama-vision-layer17-key-rope.f32" },
    { "Vcur_normed-17",      "llama-vision-layer17-value-norm.f32" },
    { "kqv_out-17",          "llama-vision-layer17-attention-context.f32" },
    { "attn_out-17",         "llama-vision-layer17-attention-output.f32" },
    { "attn_post_normed-17", "llama-vision-layer17-attention-post-norm.f32" },
    { "ffn_inp-17",          "llama-vision-layer17-attention-residual.f32" },
    { "ffn_inp_normed-17",   "llama-vision-layer17-ffn-input-norm.f32" },
    { "ffn_up-17",           "llama-vision-layer17-ffn-up.f32" },
    { "ffn_gate-17",         "llama-vision-layer17-ffn-gate.f32" },
    { "ffn_geglu_quick-17",  "llama-vision-layer17-ffn-activated.f32" },
    { "ffn_out-17",          "llama-vision-layer17-ffn-output.f32" },
    { "ffn_post_normed-17",  "llama-vision-layer17-ffn-post-norm.f32" },
};

static void quiet_log(enum ggml_log_level, const char *, void *) {}

static bool write_tensor(const std::string & path, ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr, "%s is not contiguous float32\n", tensor->name);
        return false;
    }
    std::vector<unsigned char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
    return std::fclose(file) == 0 && ok;
}

static bool capture_callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto * state = static_cast<capture_state *>(opaque);
    const char * file = nullptr;
    bool * captured = nullptr;
    if (state->trace_operations) {
        for (size_t index = 0;
             index < sizeof(operation_captures) / sizeof(operation_captures[0]);
             ++index) {
            if (std::strcmp(tensor->name, operation_captures[index].tensor) == 0) {
                if (ask) return true;
                const std::string path = state->directory + "/" +
                                         operation_captures[index].file;
                if (!write_tensor(path, tensor)) return false;
                ++state->operation_count;
                std::printf("captured %s [%lld, %lld] -> %s\n", tensor->name,
                            static_cast<long long>(tensor->ne[0]),
                            static_cast<long long>(tensor->ne[1]), path.c_str());
                return true;
            }
        }
    }
    if (std::strcmp(tensor->name, "layer_out-0") == 0) {
        file = "llama-vision-layer1.f32";
        captured = &state->layer0;
    } else if (std::strcmp(tensor->name, "layer_out-8") == 0) {
        file = "llama-vision-layer9.f32";
        captured = &state->layer8;
    } else if (std::strcmp(tensor->name, "layer_out-17") == 0) {
        file = "llama-vision-layer18.f32";
        captured = &state->layer17;
    } else if (std::strcmp(tensor->name, "layer_out-26") == 0) {
        file = "llama-vision-layer27.f32";
        captured = &state->layer26;
    } else if (std::strcmp(tensor->name, "projected") == 0) {
        file = "llama-vision-projected.f32";
        captured = &state->projected;
    }
    if (ask) return file && !*captured;
    if (!file || !captured || *captured) return true;
    const std::string path = state->directory + "/" + file;
    if (!write_tensor(path, tensor)) return false;
    *captured = true;
    if (std::strcmp(tensor->name, "projected") == 0) {
        state->projected_width = static_cast<size_t>(tensor->ne[0]);
        state->projected_tokens = static_cast<size_t>(tensor->ne[1]);
    }
    std::printf("captured %s [%lld, %lld] -> %s\n", tensor->name,
                static_cast<long long>(tensor->ne[0]),
                static_cast<long long>(tensor->ne[1]), path.c_str());
    return true;
}

static bool write_output(const std::string & path, const float * values,
                         size_t count) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const bool ok = std::fwrite(values, sizeof(float), count, file) == count;
    return std::fclose(file) == 0 && ok;
}

int main(int argc, char ** argv) {
    const bool trace_operations =
        argc == 7 && std::strcmp(argv[6], "--trace-operations") == 0;
    if (argc != 6 && !trace_operations) {
        std::fprintf(stderr,
            "usage: %s MODEL.gguf MMPROJ.gguf WIDTH HEIGHT OUTPUT_DIR "
            "[--trace-operations]\n",
            argv[0]);
        return 2;
    }
    char * end = nullptr;
    const unsigned long width_long = std::strtoul(argv[3], &end, 10);
    if (!end || *end || width_long == 0 || width_long > UINT32_MAX) return 2;
    const uint32_t width = static_cast<uint32_t>(width_long);
    const unsigned long height_long = std::strtoul(argv[4], &end, 10);
    if (!end || *end || height_long == 0 || height_long > UINT32_MAX) return 2;
    const uint32_t height = static_cast<uint32_t>(height_long);
    if (static_cast<uint64_t>(width) * height > SIZE_MAX / 3U) return 2;
    std::filesystem::create_directories(argv[5]);

    llama_log_set(quiet_log, nullptr);
    mtmd_log_set(quiet_log, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = false;
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "cannot load text model\n");
        llama_backend_free();
        return 1;
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.no_perf = true;
    llama_context * text_context = llama_init_from_model(model, context_params);
    if (!text_context) {
        std::fprintf(stderr, "cannot create text context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    capture_state capture;
    capture.directory = argv[5];
    capture.trace_operations = trace_operations;
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.warmup = false;
    params.print_timings = true;
    const unsigned threads = std::thread::hardware_concurrency();
    params.n_threads = threads ? static_cast<int>(threads) : 4;
    params.cb_eval = capture_callback;
    params.cb_eval_user_data = &capture;
    mtmd_context * context = mtmd_init_from_file(argv[2], model, params);
    if (!context) {
        std::fprintf(stderr, "cannot load multimodal projector\n");
        llama_free(text_context);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    const size_t rgb_count = static_cast<size_t>(width) * height * 3U;
    std::vector<unsigned char> rgb(rgb_count);
    for (size_t index = 0; index < rgb.size(); ++index)
        rgb[index] = static_cast<unsigned char>((index * 37U + 11U) & 255U);
    mtmd_bitmap * bitmap = mtmd_bitmap_init(width, height, rgb.data());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmap_list[] = { bitmap };
    const std::string prompt =
        std::string("<bos><|turn>user\n") + mtmd_default_marker() +
        "Describe this image.<turn|>\n"
        "<|turn>model\n<|channel>thought\n<channel|>";
    mtmd_input_text text { prompt.c_str(), false, true };
    int result = 1;
    if (!bitmap || !chunks || mtmd_tokenize(
            context, chunks, &text, bitmap_list, 1) != 0) {
        std::fprintf(stderr, "llama.cpp image preprocessing failed\n");
        goto cleanup;
    }
    for (size_t index = 0; index < mtmd_input_chunks_size(chunks); ++index) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, index);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE)
            continue;
        if (mtmd_encode_chunk(context, chunk) != 0) {
            std::fprintf(stderr, "llama.cpp vision encode failed\n");
            goto cleanup;
        }
        const size_t tokens = mtmd_input_chunk_get_n_tokens(chunk);
        const size_t embedding_width = capture.projected_width;
        const float * output = mtmd_get_output_embd(context);
        const std::string output_path =
            capture.directory + "/llama-vision-output.f32";
        if (!output || !embedding_width || capture.projected_tokens != tokens ||
            !write_output(
                output_path, output, tokens * embedding_width)) {
            std::fprintf(stderr, "cannot save llama.cpp projected vectors\n");
            goto cleanup;
        }
        std::printf("output %zu x %zu -> %s\n", tokens, embedding_width,
                    output_path.c_str());
        const bool operations_complete = !trace_operations ||
            capture.operation_count ==
                sizeof(operation_captures) / sizeof(operation_captures[0]);
        result = capture.layer0 && capture.layer8 && capture.layer17 &&
                 capture.layer26 && capture.projected && operations_complete
                    ? 0 : 1;
        if (result)
            std::fprintf(stderr, "required vision tensors were not captured\n");
        break;
    }
    if (result == 0 && !trace_operations) {
        llama_pos new_position = 0;
        if (mtmd_helper_eval_chunks(context, text_context, chunks, 0, 0,
                                    512, true, &new_position) != 0) {
            std::fprintf(stderr, "llama.cpp image-conditioned decode failed\n");
            result = 1;
        } else {
            const float * logits = llama_get_logits_ith(text_context, -1);
            const int32_t vocabulary = llama_vocab_n_tokens(
                llama_model_get_vocab(model));
            const std::string logits_path =
                capture.directory + "/llama-image-logits.f32";
            if (!logits || vocabulary <= 0 || !write_output(
                    logits_path, logits, static_cast<size_t>(vocabulary))) {
                std::fprintf(stderr, "cannot save image-conditioned logits\n");
                result = 1;
            } else {
                std::printf("image logits %d -> %s\n", vocabulary,
                            logits_path.c_str());
            }
        }
    }
cleanup:
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bitmap);
    mtmd_free(context);
    llama_free(text_context);
    llama_free_model(model);
    llama_backend_free();
    return result;
}
