"""Phase 2 - the walking skeleton.

Trains YOLO11n + TG-FEM (identity mode) for a couple of epochs on a small slice
of NEU-DET. The goal is NOT accuracy - it is to prove the plumbing:

    1. TGFEM resolves from the model YAML
    2. it sits at all three pyramid levels
    3. the model trains end to end without shape or autograd errors
    4. boxes come out of the far end

Once this passes, Phase 5 only has to replace the identity forward with real
maths - the integration is already proven.

Run:
    python scripts/train_skeleton.py
    python scripts/train_skeleton.py --epochs 5 --fraction 0.2
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tgfem import register, TGFEM  # noqa: E402

DATA = REPO_ROOT / "datasets" / "neu-det-yolo" / "neu_closed.yaml"
MODEL_CFG = REPO_ROOT / "cfg" / "yolo11-tgfem.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=0.1,
                    help="fraction of the train split to use")
    ap.add_argument("--batch", type=int, default=8,
                    help="keep small: 6 GB VRAM")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    register()
    from ultralytics import YOLO
    import torch

    print("=" * 62)
    print("Phase 2 - walking skeleton")
    print("=" * 62)

    if not DATA.exists():
        raise SystemExit(f"missing {DATA}\nrun: python scripts/prepare_neu_det.py")

    model = YOLO(str(MODEL_CFG))

    # --- structural assertions before spending any GPU time ---------------
    tgfem_layers = [i for i, m in enumerate(model.model.model) if isinstance(m, TGFEM)]
    assert len(tgfem_layers) == 3, f"expected 3 TGFEM layers, found {tgfem_layers}"

    probe = torch.randn(2, 64, 32, 32)
    assert torch.equal(model.model.model[tgfem_layers[0]](probe), probe), \
        "TGFEM is not an exact identity - ablation (a) would be invalid"

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"TGFEM at layers : {tgfem_layers}")
    print(f"identity check  : PASS")
    print(f"parameters      : {n_params:,}")
    print(f"device          : {args.device}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    print()

    # --- train -----------------------------------------------------------
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=True,
        workers=2,
        project=str(REPO_ROOT / "runs"),
        name="phase2_skeleton",
        exist_ok=True,
        plots=False,
        val=True,
    )

    # --- prove boxes come out the far end --------------------------------
    # A 2-epoch run from scratch predicts nothing above the default 0.25
    # confidence threshold, so a default predict() would report zero boxes and
    # tell us nothing about the plumbing. Drop the threshold: the point here is
    # that the detection head EMITS boxes with coordinates and class ids, not
    # that they are any good yet.
    sample = next((REPO_ROOT / "datasets" / "neu-det-yolo" / "images").glob("*.jpg"))
    result = model.predict(str(sample), device=args.device, conf=0.001, verbose=False)[0]

    print()
    print("=" * 62)
    print(f"prediction on {sample.name} (conf>=0.001): {len(result.boxes)} boxes")
    for b in result.boxes[:5]:
        cls = result.names[int(b.cls)]
        print(f"  {cls:<18} conf={float(b.conf):.4f}  "
              f"xyxy={[round(v, 1) for v in b.xyxy[0].tolist()]}")
    print("=" * 62)

    if len(result.boxes) == 0:
        raise SystemExit(
            "\nPhase 2 gate: FAILED - the head emitted no boxes at all.\n"
        )
    print("\nPhase 2 gate: PASSED - the pipeline is wired end to end.")
    print("NOTE: accuracy is meaningless here (2 epochs, 143 images). The gate")
    print("      is structural only: TG-FEM is in the graph and boxes come out.\n")


if __name__ == "__main__":
    main()
