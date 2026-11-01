/* deepseek_v4_bank_pair.h — decision logic for the double-buffered prefill
 * expert bank (COLI_CUDA_MOE_DOUBLE=1), and the VRAM reserve decode keeps
 * free for the bank(s).
 *
 * While the GPU computes layer L's chunks from the active bank, a worker
 * thread loads layer L+1's COMPLETE expert set into the other bank over the
 * device's aux stream — the transfer starts before L+1's routing is known,
 * which is exactly what loading the full layer buys. On the layer switch the
 * banks swap roles. If the prefetched layer does not match (segment restart,
 * bootstrap, worker failure) the engine falls back to the route-aware
 * on-demand refill it uses today; a PARTIAL prefetch still helps, because
 * the swap carries the per-expert valid map and the existing refill tops up
 * only the holes.
 *
 * This header holds the pure decisions so the sequencing — where
 * double-buffer bugs actually live — is unit-tested on machines with no GPU;
 * the CUDA glue in deepseek_v4.c wires threads and streams around it. */
#ifndef DEEPSEEK_V4_BANK_PAIR_H
#define DEEPSEEK_V4_BANK_PAIR_H

typedef enum {
    V4_BANK_SWAP = 0,        /* other bank holds this layer: swap roles */
    V4_BANK_LEGACY = 1,      /* refill the active bank on demand */
} V4BankPairAction;

/* What to do when a chunk arrives for `incoming_layer`. `other_layer` is the
 * layer the inactive bank was prefetched for (-1 = none/failed), valid only
 * once the caller has JOINED the prefetch worker — deciding on a bank whose
 * upload is still in flight is the classic double-buffer bug, so the caller
 * must join first and only then consult this. */
static inline V4BankPairAction coli_v4_bank_pair_decide(int double_on,
                                                        int other_layer,
                                                        int incoming_layer) {
    if (!double_on) return V4_BANK_LEGACY;
    if (other_layer < 0 || other_layer != incoming_layer)
        return V4_BANK_LEGACY;
    return V4_BANK_SWAP;
}

/* Which layer to prefetch while `current_layer` computes: the next one, or
 * -1 when there is none (last layer) or the feature is off. Prefetching
 * anything OTHER than current+1 is never right: target_batch walks layers in
 * ascending order and a segment restart goes back to 0, which the mismatch
 * path above absorbs. */
static inline int coli_v4_bank_pair_prefetch_target(int double_on,
                                                    int current_layer,
                                                    int num_layers) {
    if (!double_on || current_layer < 0) return -1;
    if (current_layer + 1 >= num_layers) return -1;
    return current_layer + 1;
}

/* ---- VRAM the decode mirror cache leaves free for the bank(s) ----
 * Decode grows its routed-expert mirror cache while free VRAM stays above a
 * reserve, and the next large prefill allocates its bank(s) from what is
 * left, so the default reserve is sized from the bank itself: attention/work
 * headroom (the whole reserve without the bank), plus `banks` x
 * `bank_bytes` (two with COLI_CUDA_MOE_DOUBLE until its prefetch has
 * latched off; a second bank that did not fit is retried by the next
 * prefill, so it stays reserved), plus COLI_V4_MIRROR_PROBE_EVERY mirrors of
 * `mirror_bytes`: the guard re-reads free VRAM only every that many misses,
 * so the cache can grow that far past the reserve before it notices. A card
 * too small for that after the dense mirrors (two 256-slot banks alone are
 * 6.4 GiB) would freeze the cache below the 8 live entries GPU decode needs
 * and run every expert on the CPU, so below COLI_V4_MIRROR_MIN_LIVE entries
 * the default falls back to the headroom: decode keeps a working mirror set,
 * and a bank that no longer fits falls back to the CPU union.
 * `bank_bytes` <= 0 = no bank. Returns MiB, rounded up. */
#define COLI_V4_RESERVE_HEADROOM_MB 600
#define COLI_V4_MIRROR_PROBE_EVERY 64
#define COLI_V4_MIRROR_MIN_LIVE 32

static inline long long coli_v4_mirror_reserve_mb(long long bank_bytes,
                                                  int banks,
                                                  long long mirror_bytes,
                                                  int live) {
    if (bank_bytes <= 0 || banks < 1 || live < COLI_V4_MIRROR_MIN_LIVE)
        return COLI_V4_RESERVE_HEADROOM_MB;
    long long bytes = (long long)banks * bank_bytes +
                      (long long)COLI_V4_MIRROR_PROBE_EVERY * mirror_bytes;
    return COLI_V4_RESERVE_HEADROOM_MB + ((bytes + (1LL << 20) - 1) >> 20);
}

#endif /* DEEPSEEK_V4_BANK_PAIR_H */
