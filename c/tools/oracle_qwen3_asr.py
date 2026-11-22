#!/usr/bin/env python3
"""Generate small oracle fixtures for Qwen3-ASR against HuggingFace transformers."""
import json, os, wave, numpy as np, torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor

MODEL = os.environ.get("QWEN3_ASR_MODEL", "/home/dorin/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5")
WAV = os.environ.get("WAV", "/tmp/opencode/bench_wavs/hello.wav")
OUT = os.environ.get("OUT", "/media/dorin/more_data/work/colibri/c/testdata/qwen3_asr_hello")
os.makedirs(OUT, exist_ok=True)

with wave.open(WAV, 'rb') as wf:
    sr = wf.getframerate(); n = wf.getnframes(); ch = wf.getnchannels()
    data = wf.readframes(n)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1: audio = audio.reshape(-1, ch).mean(1)
if sr != 16000:
    n_out = int(len(audio) * 16000 / sr)
    idx = np.arange(n_out) * sr / 16000
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0+1, len(audio)-1)
    frac = idx - i0
    audio = audio[i0]*(1-frac) + audio[i1]*frac
    sr = 16000

processor = AutoProcessor.from_pretrained(MODEL)
fe = Qwen3ASRFeatureExtractor()
mel_out = fe(audio, sampling_rate=sr, return_tensors='np', return_attention_mask=True)
np.save(os.path.join(OUT, 'mel.npy'), mel_out['input_features'])
np.save(os.path.join(OUT, 'mel_attention_mask.npy'), mel_out['attention_mask'])
print('mel', mel_out['input_features'].shape, 'mask_sum', int(mel_out['attention_mask'].sum()))

print('loading model...')
from transformers import AutoConfig
from safetensors.torch import load_file
cfg = AutoConfig.from_pretrained(MODEL)
cfg.audio_config.output_dim = cfg.text_config.hidden_size
model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16)
idx=json.load(open(os.path.join(MODEL,'model.safetensors.index.json')))
sd={}
for shard in sorted(set(idx['weight_map'].values())):
    sd.update(load_file(os.path.join(MODEL,shard)))
remap={}
for k,v in sd.items():
    nk=k
    if nk.startswith('thinker.lm_head.'): nk = nk[len('thinker.'):]
    elif nk.startswith('thinker.model.'): nk = 'model.language_model.' + nk[len('thinker.model.'):]
    elif nk.startswith('thinker.audio_tower.proj1.'): nk = 'model.multi_modal_projector.linear_1.' + nk[len('thinker.audio_tower.proj1.'):]
    elif nk.startswith('thinker.audio_tower.proj2.'): nk = 'model.multi_modal_projector.linear_2.' + nk[len('thinker.audio_tower.proj2.'):]
    elif nk.startswith('thinker.audio_tower.'): nk = 'model.audio_tower.' + nk[len('thinker.audio_tower.'):]
    elif nk.startswith('thinker.'): nk = 'model.' + nk[len('thinker.'):]
    remap[nk]=v
missing, unexpected = model.load_state_dict(remap, strict=False)
print('remap missing', len(missing), 'unexpected', len(unexpected))
model.eval()
print('model loaded')

tok = processor.tokenizer
input_features = torch.from_numpy(mel_out['input_features']).to(torch.bfloat16)
frame_mask = torch.from_numpy(mel_out['attention_mask']).to(torch.long)

with torch.no_grad():
    audio_out = model.get_audio_features(input_features=input_features, input_features_mask=frame_mask)
    audio_emb = audio_out.pooler_output
    print('audio_emb', audio_emb.shape, audio_emb.dtype)
np.save(os.path.join(OUT, 'audio_emb.npy'), audio_emb.float().cpu().numpy())

n_audio = audio_emb.shape[0]
audio_pad_id = tok.convert_tokens_to_ids('<|audio_pad|>')
audio_start_id = tok.convert_tokens_to_ids('<|audio_start|>')
audio_end_id = tok.convert_tokens_to_ids('<|audio_end|>')
im_start = tok.convert_tokens_to_ids('<|im_start|>')
im_end = tok.convert_tokens_to_ids('<|im_end|>')
nl = tok.encode('\n', add_special_tokens=False)

prefill_text = 'language English<asr_text>'
prefill_ids = tok.encode(prefill_text, add_special_tokens=False)
print('prefill_ids', prefill_ids)

prompt_ids = []
prompt_ids += [im_start] + tok.encode('system', add_special_tokens=False) + nl + [im_end] + nl
prompt_ids += [im_start] + tok.encode('user', add_special_tokens=False) + nl
prompt_ids += [audio_start_id] + [audio_pad_id] * n_audio + [audio_end_id] + [im_end] + nl
prompt_ids += [im_start] + tok.encode('assistant', add_special_tokens=False) + nl
prompt_ids += prefill_ids
print('prompt_len', len(prompt_ids), 'n_audio', n_audio)
np.save(os.path.join(OUT, 'prompt_ids.npy'), np.array(prompt_ids, dtype=np.int32))

input_ids = torch.tensor([prompt_ids], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)
with torch.no_grad():
    inputs_embeds = model.get_input_embeddings()(input_ids)
    audio_mask = (input_ids == audio_pad_id).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_emb.to(inputs_embeds.dtype))

    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[0, -1, :].float().cpu().numpy()
    topk_idx = np.argsort(logits)[-10:][::-1]
    print('top10', [(int(i), float(logits[i]), tok.decode([int(i)])) for i in topk_idx])
    np.save(os.path.join(OUT, 'step0_logits_topk_idx.npy'), topk_idx.astype(np.int32))
    np.save(os.path.join(OUT, 'step0_logits_topk_val.npy'), logits[topk_idx].astype(np.float32))
    np.save(os.path.join(OUT, 'step0_argmax.npy'), np.array([int(logits.argmax())], dtype=np.int32))

    gen = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=32, do_sample=False)
    new_ids = gen[0].tolist()
    print('new_ids', new_ids[:20])
    text = tok.decode(new_ids, skip_special_tokens=False)
    print('decoded', text)
    text_clean = tok.decode(new_ids, skip_special_tokens=True)
    print('text', text_clean)
np.save(os.path.join(OUT, 'greedy_ids.npy'), np.array(new_ids, dtype=np.int32))
with open(os.path.join(OUT, 'transcript.txt'), 'w') as f: f.write(text_clean)

meta = {'model': MODEL, 'wav': WAV, 'sr': sr, 'n_samples': int(len(audio)), 'n_audio_tokens': int(n_audio), 'prompt_len': len(prompt_ids)}
with open(os.path.join(OUT, 'meta.json'), 'w') as f: json.dump(meta, f, indent=2)
print('done')
