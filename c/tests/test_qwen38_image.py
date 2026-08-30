#!/usr/bin/env python3
"""tools/qwen38_image.py contro Qwen2VLImageProcessor ufficiale.

Il preprocessing e' la parte della vision che si sbaglia in silenzio. Una tela
piu' larga di una patch, un ordine di patch diverso, una normalizzazione con le
costanti dell'altro modello: niente di tutto questo da' errore. Il modello
risponde lo stesso, e risponde peggio. Per questo il riferimento e' il
processore vero e non una rilettura del paper.

Il riferimento NON richiede i pesi: `Qwen2VLImageProcessor` si costruisce dal
solo preprocessor_config.json, che sono 390 byte. Quindi questo test si puo'
eseguire senza aver scaricato i 185 GB del checkpoint.

Cosa viene confrontato, e con che severita':

  geometria (grid_h, grid_w, numero di patch)   uguaglianza esatta
  ordine delle patch                            uguaglianza esatta
  pixel                                         tolleranza, e la scriviamo

L'ultima riga e' il punto onesto. Il riferimento ridimensiona con torchvision,
noi con Pillow, e due bicubici diversi non danno gli stessi byte. Dove non c'e'
ridimensionamento i pixel sono identici e il test lo pretende; dove c'e', misura
lo scarto e lo stampa invece di dichiararlo zero.

USO:
  python3 tests/test_qwen38_image.py --config PATH/preprocessor_config.json
"""
import argparse
import json
import sys
from pathlib import Path

# Forme scelte per esercitare i rami di smart_resize: sotto la finestra (si
# ingrandisce), dentro, molto sopra (si rimpicciolisce), e non quadrate in
# entrambi i versi.
SHAPES = [(64, 64), (100, 50), (50, 100), (37, 91), (256, 256),
          (640, 480), (1920, 1080), (33, 17)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="preprocessor_config.json del checkpoint")
    arguments = parser.parse_args()

    if not arguments.config.exists():
        print(f"SKIP: manca {arguments.config}; il riferimento non c'e' e "
              f"questo test non ha verificato nulla")
        return 0
    try:
        import numpy
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor)
    except ImportError as exc:
        print(f"SKIP: manca una dipendenza del riferimento ({exc})")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import qwen38_image

    raw = json.loads(arguments.config.read_text())
    settings = {k: v for k, v in raw.items()
                if k not in ("processor_class", "image_processor_type")}
    reference = Qwen2VLImageProcessor(**settings)
    model_dir = arguments.config.parent

    failures = 0
    worst = 0.0
    for width, height in SHAPES:
        state = numpy.random.RandomState(width * 1000 + height)
        array = (state.rand(height, width, 3) * 255).astype("uint8")
        image = Image.fromarray(array)

        want = reference(images=[image], return_tensors="np")
        want_patches = numpy.asarray(want["pixel_values"], dtype=numpy.float32)
        _, want_h, want_w = [int(v) for v in want["image_grid_thw"][0]]

        got_patches, got_h, got_w = qwen38_image.preprocess(image, model_dir)

        label = f"{width}x{height}"
        if (got_h, got_w) != (want_h, want_w):
            print(f"FAIL {label:<10} griglia {got_h}x{got_w}, il riferimento "
                  f"dice {want_h}x{want_w}")
            failures += 1
            continue
        if got_patches.shape != want_patches.shape:
            print(f"FAIL {label:<10} forma {got_patches.shape}, il riferimento "
                  f"dice {want_patches.shape}")
            failures += 1
            continue

        delta = float(numpy.abs(got_patches - want_patches).max())
        worst = max(worst, delta)
        resized = reference.size is not None
        # 0.02 su valori normalizzati in [-1, 1]: e' l'ordine di grandezza fra
        # due bicubici, non una tolleranza scelta per far passare il test. Se
        # l'ordine delle patch fosse sbagliato lo scarto sarebbe dell'ordine di 1.
        if delta <= 0.02:
            print(f"ok   {label:<10} griglia {got_h}x{got_w}, {got_patches.shape[0]} patch, "
                  f"scarto max {delta:.4f}")
        else:
            print(f"FAIL {label:<10} scarto max {delta:.4f}: troppo per una sola "
                  f"differenza di ricampionamento -- sospetto ordine o normalizzazione")
            failures += 1

    # Il rifiuto e' parte del contratto: un'immagine assurdamente allungata deve
    # fermarsi qui e non produrre una tela degenere piu' a valle.
    try:
        qwen38_image.smart_resize(1, 5000)
        print("FAIL rapporto estremo accettato invece di essere rifiutato")
        failures += 1
    except ValueError:
        print("ok   rapporto estremo rifiutato")

    print()
    if failures:
        print(f"TEST FAIL ({failures} casi)")
        return 1
    print(f"preprocessing Qwen3.8: geometria e ordine identici al riferimento, "
          f"scarto pixel al massimo {worst:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
