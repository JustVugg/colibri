/* Owned by one attention state; bounded by the speculative batch, never by
 * context length. The base snapshot remains the authoritative rollback. */
struct ColiV4AttentionTrial {
    int start, batch, recorded, hidden, head_dim, comp_rows, idx_rows;
    float *values;
};

void coli_v4_attention_trial_discard(ColiDeepSeekV4WindowAttentionState *state) {
    if (!state || !state->trial) return;
    free(state->trial->values);
    free(state->trial);
    state->trial = NULL;
}

int coli_v4_attention_trial_begin(ColiDeepSeekV4WindowAttentionState *state,
                                  int start, int batch) {
    if (!state || start < 0 || batch < 1 || batch > 25) return -1;
    coli_v4_attention_trial_discard(state);
    state->trial = calloc(1, sizeof(*state->trial));
    if (!state->trial) return -1;
    state->trial->start = start;
    state->trial->batch = batch;
    return 0;
}

int coli_v4_attention_trial_ready(const ColiDeepSeekV4WindowAttentionState *state) {
    return state && state->trial && state->trial->values &&
           state->trial->recorded == state->trial->batch;
}

static void v4_attention_trial_record(ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4Config *config, const float *inputs, const float *kv,
    int start, int batch, const float *comp_kv, const float *comp_gate,
    int comp_rows, const float *idx_kv, const float *idx_gate, int idx_rows) {
    struct ColiV4AttentionTrial *trial = state->trial;
    if (!trial) return;
    /* Missing GPU projections use the established full replay fallback. A
     * second projection could use different batch arithmetic, so never
     * substitute it for the values the verifier actually consumed. */
    if (start != trial->start + trial->recorded || batch > trial->batch - trial->recorded ||
        (state->compressor && (!comp_kv || !comp_gate)) ||
        (state->indexer && (!idx_kv || !idx_gate))) {
        coli_v4_attention_trial_discard(state);
        return;
    }
    if (!trial->values) {
        trial->hidden = config->hidden_size;
        trial->head_dim = config->head_dim;
        trial->comp_rows = comp_rows;
        trial->idx_rows = idx_rows;
        size_t width = (size_t)trial->hidden + trial->head_dim + 2 * comp_rows + 2 * idx_rows;
        trial->values = malloc((size_t)trial->batch * width * sizeof(float));
        if (!trial->values) { coli_v4_attention_trial_discard(state); return; }
    }
    if (trial->comp_rows != comp_rows || trial->idx_rows != idx_rows) {
        coli_v4_attention_trial_discard(state);
        return;
    }
    const float *sources[] = {inputs, kv, comp_kv, comp_gate, idx_kv, idx_gate};
    int widths[] = {trial->hidden, trial->head_dim, comp_rows, comp_rows, idx_rows, idx_rows};
    float *dst = trial->values;
    for (int field = 0; field < 6; field++) {
        if (widths[field])
            memcpy(dst + (size_t)trial->recorded * widths[field], sources[field],
                   (size_t)batch * widths[field] * sizeof(float));
        dst += (size_t)trial->batch * widths[field];
    }
    trial->recorded += batch;
}

/* Call only after restoring the base snapshot. All attention projections and
 * transformer activations remain those of the original verifier batch. */
int coli_v4_attention_trial_retain(ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights, const ColiDeepSeekV4Config *config,
    int retained, char *error, size_t error_size) {
    if (!coli_v4_attention_trial_ready(state) || retained < 0 || retained > state->trial->batch)
        return set_error(error, error_size, "invalid speculative attention prefix");
    struct ColiV4AttentionTrial *t = state->trial;
    const float *inputs = t->values;
    const float *kv = inputs + (size_t)t->batch * t->hidden;
    const float *comp_kv = kv + (size_t)t->batch * t->head_dim;
    const float *comp_gate = comp_kv + (size_t)t->batch * t->comp_rows;
    const float *idx_kv = comp_gate + (size_t)t->batch * t->comp_rows;
    const float *idx_gate = idx_kv + (size_t)t->batch * t->idx_rows;
    if (prepare_compressed_state(state, weights, config, error, error_size)) return -1;
    for (int item = 0; item < retained; item++) {
        int position = t->start + item;
        if (state->compressor) {
            int produced = 0;
            if (grow_compressed_state(state, error, error_size) ||
                coli_v4_compressor_advance(state->compressor,
                    state->compressed + (size_t)state->compressed_count * t->head_dim,
                    &produced, comp_kv + (size_t)item * t->comp_rows,
                    comp_gate + (size_t)item * t->comp_rows, position, error, error_size)) return -1;
            state->compressed_count += produced;
        }
        if (state->indexer && coli_v4_indexer_advance(state->indexer,
                inputs + (size_t)item * t->hidden, position,
                idx_kv + (size_t)item * t->idx_rows,
                idx_gate + (size_t)item * t->idx_rows, error, error_size) < 0) return -1;
        memcpy(state->kv + (size_t)(position % state->window_size) * t->head_dim,
               kv + (size_t)item * t->head_dim, (size_t)t->head_dim * sizeof(float));
    }
    return 0;
}
