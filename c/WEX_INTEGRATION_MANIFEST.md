# WEX Integration Manifest

## Files added
- expert_store_wex.h
- expert_store_wex.c
- wex_format.h
- wex_neural_rans.h
- wex_palette.h
- wex4.h
- wex_decode.h
- wex_decode.c
- qwen38_wex.h
- tools/repack_v4_wex.py
- tools/repack_q38_mxfp4_wex.py
- tools/build_nr3.py

## Makefile changes
- Makefile.deepseek-v4: WEX_OBJS added, V4_OBJS updated
- Makefile: colibri link line includes expert_store_wex.o wex_decode.o

## Docs
- docs/INTEGRATION-WEX.md

## Env vars
COLI_EXPERT_STORE=wex
COLI_WEX_PATH
COLI_WEX_ROOT
COLI_WEX_VERIFY
COLI_WEX_GPU_DIRECT
V4_WEX_BATCH
V4_WEX_SUBREADS
