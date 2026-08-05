#!/usr/bin/env python3
"""Calibrated KV-transfer model for layer streaming and phase splitting."""

import argparse
import json


def transfer_finish(prefill_s, layers, bytes_per_layer, bandwidth_Bps):
    network_done = 0.0
    transfer_s = bytes_per_layer / bandwidth_Bps
    for layer in range(1, layers + 1):
        produced = prefill_s * layer / layers
        network_done = max(network_done, produced) + transfer_s
    return network_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=501)
    ap.add_argument("--layers", type=int, default=79)
    ap.add_argument("--kv-latent", type=int, default=512)
    ap.add_argument("--kv-rope", type=int, default=64)
    ap.add_argument("--index-dim", type=int, default=128)
    ap.add_argument("--prefill-s", type=float, default=44.198862534)
    ap.add_argument(
        "--bandwidth-gbps",
        type=float,
        nargs="+",
        default=(1, 10, 25, 50, 100, 200),
        help="Decimal link rates in gigabits/s.",
    )
    args = ap.parse_args()

    row_bytes = (args.kv_latent + args.kv_rope + args.index_dim) * 4
    bytes_per_layer = args.tokens * row_bytes
    total_bytes = args.layers * bytes_per_layer
    rows = []
    for gbps in args.bandwidth_gbps:
        bandwidth = gbps * 1e9 / 8
        whole_s = total_bytes / bandwidth
        streamed_done = transfer_finish(
            args.prefill_s, args.layers, bytes_per_layer, bandwidth
        )
        visible_s = max(0.0, streamed_done - args.prefill_s)
        rows.append(
            {
                "bandwidth_gbps": gbps,
                "whole_kv_transfer_ms": whole_s * 1000,
                "layer_stream_visible_ms": visible_s * 1000,
                "hidden_fraction": 1 - visible_s / whole_s,
                "minimum_phase_specialization_gain_ms": visible_s * 1000,
            }
        )
    print(
        json.dumps(
            {
                "inputs": vars(args),
                "row_bytes_per_layer_token": row_bytes,
                "total_kv_bytes": total_bytes,
                "results": rows,
                "model_scope": (
                    "Uniform per-layer production and one serialized transfer link; "
                    "excludes RPC, queueing, serialization, and remote allocation."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
