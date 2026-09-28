#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct capture_state {
    std::string directory;
    bool captured_input = false;
    bool captured_attention = false;
    bool captured_query = false;
    bool captured_key = false;
    bool captured_value = false;
    bool captured_dense_input = false;
    bool captured_router = false;
    bool captured_layer = false;
    bool captured_final_residual = false;
    bool captured_final_normalized = false;
    bool captured_logits = false;
};

static bool write_tensor(const std::string & path, ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr, "oracle tensor %s is not contiguous F32\n",
                     tensor->name);
        return false;
    }
    std::vector<unsigned char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) {
        std::fprintf(stderr, "cannot create %s\n", path.c_str());
        return false;
    }
    bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size() &&
              std::fclose(file) == 0;
    if (!ok) std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return ok;
}

static bool write_last_vector(const std::string & path, ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        std::fprintf(stderr, "oracle tensor %s is not contiguous F32\n",
                     tensor->name);
        return false;
    }
    const size_t bytes = static_cast<size_t>(tensor->ne[0]) * sizeof(float);
    const size_t offset = static_cast<size_t>(tensor->ne[1] - 1) * bytes;
    std::vector<unsigned char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), offset, data.size());
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    return std::fwrite(data.data(), 1, data.size(), file) == data.size() &&
           std::fclose(file) == 0;
}

static bool capture_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    const char * file_name = nullptr;
    bool * captured = nullptr;
    bool last_vector = false;
    if (std::strcmp(tensor->name, "inp_scaled") == 0) {
        file_name = "llama-input.f32"; captured = &state->captured_input;
    } else if (std::strcmp(tensor->name, "attn_post_norm-0") == 0) {
        file_name = "llama-attention-postnorm.f32"; captured = &state->captured_attention;
    } else if (std::strcmp(tensor->name, "Qcur_pos-0") == 0) {
        file_name = "llama-query.f32"; captured = &state->captured_query;
    } else if (std::strcmp(tensor->name, "Kcur_pos-0") == 0) {
        file_name = "llama-key.f32"; captured = &state->captured_key;
    } else if (std::strcmp(tensor->name, "Vcur_normed-0") == 0) {
        file_name = "llama-value.f32"; captured = &state->captured_value;
    } else if (std::strcmp(tensor->name, "ffn_norm_1-0") == 0) {
        file_name = "llama-dense-input.f32"; captured = &state->captured_dense_input;
    } else if (std::strcmp(tensor->name, "ffn_moe_logits-0") == 0) {
        file_name = "llama-router-logits.f32"; captured = &state->captured_router;
    } else if (std::strcmp(tensor->name, "l_out-0") == 0) {
        file_name = "llama-layer0.f32"; captured = &state->captured_layer;
    } else if (std::strcmp(tensor->name, "l_out-29") == 0) {
        file_name = "llama-final-residual.f32";
        captured = &state->captured_final_residual; last_vector = true;
    } else if (std::strcmp(tensor->name, "result_norm") == 0) {
        file_name = "llama-final-normalized.f32";
        captured = &state->captured_final_normalized; last_vector = true;
    } else if (std::strcmp(tensor->name, "result_output") == 0) {
        file_name = "llama-logits.f32";
        captured = &state->captured_logits; last_vector = true;
    }
    if (ask) return file_name != nullptr && (captured == nullptr || !*captured);
    if (!file_name || !captured || *captured) return true;
    const std::string path = state->directory + "/" + file_name;
    if (!(last_vector ? write_last_vector(path, tensor) :
                        write_tensor(path, tensor))) return false;
    *captured = true;
    std::printf("captured %-10s shape=%lld x %lld -> %s\n", tensor->name,
                static_cast<long long>(tensor->ne[0]),
                static_cast<long long>(tensor->ne[1]), path.c_str());
    return true;
}

static bool write_tokens(const std::string & path,
                         const std::vector<llama_token> & tokens) {
    FILE * file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    for (size_t index = 0; index < tokens.size(); ++index)
        std::fprintf(file, "%s%d", index ? " " : "", tokens[index]);
    std::fputc('\n', file);
    return std::fclose(file) == 0;
}

static int hex_digit(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

int main(int argc, char ** argv) {
    if (argc == 5 && (std::strcmp(argv[1], "--tokenize") == 0 ||
                      std::strcmp(argv[1], "--tokenize-special") == 0 ||
                      std::strcmp(argv[1], "--tokenize-special-hex") == 0 ||
                      std::strcmp(argv[1], "--tokenize-file") == 0 ||
                      std::strcmp(argv[1], "--tokenize-special-file") == 0)) {
        const char * model_path = argv[2];
        const char * output_path = argv[4];
        std::vector<char> prompt_storage;
        const char * prompt = argv[3];
        size_t prompt_length = std::strlen(prompt);
        const bool parse_special =
            std::strcmp(argv[1], "--tokenize-special") == 0 ||
            std::strcmp(argv[1], "--tokenize-special-hex") == 0 ||
            std::strcmp(argv[1], "--tokenize-special-file") == 0;
        const bool add_special = !parse_special;
        if (std::strcmp(argv[1], "--tokenize-special-hex") == 0) {
            const size_t hex_length = std::strlen(argv[3]);
            if (hex_length % 2 != 0) {
                std::fprintf(stderr, "hex tokenizer input has odd length\n");
                return 1;
            }
            prompt_storage.resize(hex_length / 2);
            for (size_t index = 0; index < prompt_storage.size(); ++index) {
                const int high = hex_digit(argv[3][index * 2]);
                const int low = hex_digit(argv[3][index * 2 + 1]);
                if (high < 0 || low < 0) {
                    std::fprintf(stderr, "invalid hex tokenizer input\n");
                    return 1;
                }
                prompt_storage[index] = static_cast<char>((high << 4) | low);
            }
            prompt = prompt_storage.data();
            prompt_length = prompt_storage.size();
        } else if (std::strcmp(argv[1], "--tokenize") != 0 &&
            std::strcmp(argv[1], "--tokenize-special") != 0) {
            FILE * input = std::fopen(argv[3], "rb");
            if (!input || std::fseek(input, 0, SEEK_END) != 0) {
                if (input) std::fclose(input);
                std::fprintf(stderr, "cannot read tokenizer input file\n");
                return 1;
            }
            long length = std::ftell(input);
            if (length < 0 || std::fseek(input, 0, SEEK_SET) != 0) {
                std::fclose(input);
                std::fprintf(stderr, "cannot size tokenizer input file\n");
                return 1;
            }
            prompt_storage.resize(static_cast<size_t>(length));
            if ((length && std::fread(prompt_storage.data(), 1,
                                      prompt_storage.size(), input) !=
                           prompt_storage.size()) || std::fclose(input) != 0) {
                std::fprintf(stderr, "cannot read tokenizer input file\n");
                return 1;
            }
            prompt = prompt_storage.data();
            prompt_length = prompt_storage.size();
        }
        llama_backend_init();
        ggml_backend_dev_t devices[] = {
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr
        };
        llama_model_params params = llama_model_default_params();
        params.devices = devices;
        params.n_gpu_layers = 0;
        params.vocab_only = true;
        llama_model * model = llama_model_load_from_file(model_path, params);
        if (!model) {
            std::fprintf(stderr, "cannot load llama.cpp tokenizer model\n");
            llama_backend_free();
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        int32_t count = llama_tokenize(vocab, prompt,
                                       static_cast<int32_t>(prompt_length),
                                       nullptr, 0, add_special, parse_special);
        if (count >= 0) {
            std::fprintf(stderr, "llama.cpp tokenization failed\n");
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, prompt,
                               static_cast<int32_t>(prompt_length),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special, parse_special);
        if (count <= 0) {
            std::fprintf(stderr, "llama.cpp tokenization failed\n");
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        tokens.resize(static_cast<size_t>(count));
        if (!write_tokens(output_path, tokens)) {
            std::fprintf(stderr, "cannot write llama.cpp tokens\n");
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        std::printf("tokens: %d -> %s\n", count, output_path);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }
    const bool prompt_special_file =
        argc == 5 && std::strcmp(argv[1], "--prompt-special-file") == 0;
    if ((!prompt_special_file && argc != 4) ||
        (prompt_special_file && argc != 5)) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf PROMPT OUTPUT_DIR\n"
                     "       %s --prompt-special-file MODEL.gguf INPUT OUTPUT_DIR\n"
                     "       %s --tokenize MODEL.gguf PROMPT OUTPUT_FILE\n"
                     "       %s --tokenize-special MODEL.gguf PROMPT OUTPUT_FILE\n"
                     "       %s --tokenize-special-hex MODEL.gguf HEX OUTPUT_FILE\n"
                     "       %s --tokenize-file MODEL.gguf INPUT OUTPUT_FILE\n"
                     "       %s --tokenize-special-file MODEL.gguf INPUT OUTPUT_FILE\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0],
                     argv[0]);
        return 2;
    }
    const char * model_path = prompt_special_file ? argv[2] : argv[1];
    const char * prompt = prompt_special_file ? nullptr : argv[2];
    std::vector<char> full_prompt_storage;
    size_t full_prompt_length = prompt ? std::strlen(prompt) : 0;
    capture_state capture{prompt_special_file ? argv[4] : argv[3]};
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    int status = 1;
    if (prompt_special_file) {
        FILE * input = std::fopen(argv[3], "rb");
        if (!input || std::fseek(input, 0, SEEK_END) != 0) {
            if (input) std::fclose(input);
            std::fprintf(stderr, "cannot read full prompt input file\n");
            return 1;
        }
        const long length = std::ftell(input);
        if (length < 0 || std::fseek(input, 0, SEEK_SET) != 0) {
            std::fclose(input);
            return 1;
        }
        full_prompt_storage.resize(static_cast<size_t>(length));
        if ((length && std::fread(full_prompt_storage.data(), 1,
                                  full_prompt_storage.size(), input) !=
                       full_prompt_storage.size()) || std::fclose(input) != 0) {
            std::fprintf(stderr, "cannot read full prompt input file\n");
            return 1;
        }
        prompt = full_prompt_storage.data();
        full_prompt_length = full_prompt_storage.size();
    }
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_context_params context_params = llama_context_default_params();
    ggml_backend_dev_t devices[] = {
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr
    };
    model_params.devices = devices;
    model_params.n_gpu_layers = 0;
    model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "cannot load llama.cpp reference model\n");
        goto cleanup;
    }
    context_params.n_ctx = 32;
    context_params.n_batch = 32;
    context_params.n_ubatch = 32;
    context_params.n_threads = 16;
    context_params.n_threads_batch = 16;
    context_params.cb_eval = capture_callback;
    context_params.cb_eval_user_data = &capture;
    context_params.offload_kqv = false;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context = llama_init_from_model(model, context_params);
    if (!context) {
        std::fprintf(stderr, "cannot create llama.cpp reference context\n");
        goto cleanup;
    }

    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_special = !prompt_special_file;
        const bool parse_special = prompt_special_file;
        int32_t count = llama_tokenize(vocab, prompt,
                                       static_cast<int32_t>(full_prompt_length),
                                       nullptr, 0, add_special, parse_special);
        if (count >= 0) {
            std::fprintf(stderr, "tokenizer did not report required capacity\n");
            goto cleanup;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(-count));
        count = llama_tokenize(vocab, prompt,
                               static_cast<int32_t>(full_prompt_length),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special, parse_special);
        if (count <= 0) {
            std::fprintf(stderr, "prompt tokenization failed\n");
            goto cleanup;
        }
        tokens.resize(static_cast<size_t>(count));
        if (!write_tokens(capture.directory + "/llama-tokens.txt", tokens)) {
            std::fprintf(stderr, "cannot write reference token IDs\n");
            goto cleanup;
        }
        llama_batch batch = llama_batch_get_one(tokens.data(), count);
        if (llama_decode(context, batch) != 0) {
            std::fprintf(stderr, "llama.cpp reference decode failed\n");
            goto cleanup;
        }
        std::printf("tokens: %d\n", count);
    }
    if (!capture.captured_input || !capture.captured_query ||
        !capture.captured_key || !capture.captured_value ||
        !capture.captured_attention || !capture.captured_dense_input ||
        !capture.captured_router || !capture.captured_layer ||
        !capture.captured_final_residual ||
        !capture.captured_final_normalized || !capture.captured_logits) {
        std::fprintf(stderr, "required llama.cpp tensors were not captured\n");
        goto cleanup;
    }
    status = 0;

cleanup:
    if (context) llama_free(context);
    if (model) llama_free_model(model);
    llama_backend_free();
    return status;
}
