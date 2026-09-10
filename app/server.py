"""Local web app: pick a model, upload an image, type a description, get boxes.

Runs entirely on your own machine - no deployment, no network dependency during
a demo. Inference lives in `tgfem.inference.DefectDetector`; this file owns HTTP
and nothing else, so the same detector can back a deployed service later without
being rewritten.

Which models are loaded, and why these
--------------------------------------
Merging is not always bad and not always good - it was measured both ways:

  * NEU + GC10 (two STEEL datasets, 16 classes) is WORSE than the specialists
    on both taxonomies: GC10 0.598 vs 0.638, NEU 0.704 vs 0.720. Near-synonymous
    classes across two steel processes make each one harder. Not shipped.
  * NEU + crack (steel + concrete, 7 classes) costs nothing: NEU 0.7264 against
    the specialist's 0.7205, crack 0.448 against 0.465 on a quarter of the crack
    training images. Shipped, and it REPLACES both specialists.

The reason `neu_crack` ships even though the crack specialist scores higher on
crack: the specialist has one class, so its contrastive head never had to
separate one text embedding from another and it returns the same boxes for
`banana` as for a real query (PHASE-9.md section 4). The 7-class model rejects
`banana`, and rejects `scratches on the steel surface` on a concrete image -
cross-class discrimination, not merely unknown-text failure. A 1.6-point mAP
difference does not outweigh a text interface that works.

The `neu` and `crack` specialists stay in runs/ as the evidence for this and can
be re-added here in one line; they are not loaded because a demo with two extra
models that answer the same questions worse is just a way to pick the wrong one.

Run:
    python app/server.py
    python app/server.py --port 8080 --only neu_crack
Then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from tgfem.inference import DefectDetector  # noqa: E402


@dataclass
class ModelSpec:
    key: str
    label: str
    blurb: str
    run: str
    detector: DefectDetector | None = field(default=None, repr=False)

    @property
    def checkpoint(self) -> Path:
        return REPO_ROOT / "runs" / self.run / "weights" / "best.pt"

    @property
    def sample_dir(self) -> Path:
        return REPO_ROOT / "app" / "samples" / self.key


# Every entry is a phrase-augmented checkpoint. That choice is on EVIDENCE, not
# on mAP: on NEU the augmented model scores 0.7205 vs 0.7268 for the plain one,
# but responds to 5 of 6 phrasings instead of 2 of 6 (phase-notes/PHASE-7.md
# section 4). A demo where the user types their own words needs the robust
# model, not the one with the best number.
MODELS = [
    ModelSpec("neu_crack", "Steel + concrete", "7 classes · steel strip and concrete cracks",
              "phase6_neu_crack_closed_tgfem_phraseaug"),
    ModelSpec("gc10", "Steel · GC10-DET", "10 classes · galvanised steel sheet",
              "phase6_gc10_closed_tgfem_phraseaug"),
]

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="TG-FEM defect detection")
_loaded: dict[str, ModelSpec] = {}


def _spec(key: str) -> ModelSpec:
    spec = _loaded.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"no model named {key!r}")
    return spec


def _guarded(sample_dir: Path, name: str) -> Path:
    """Resolve a sample name inside its model's folder. A name like
    "../../secrets" must not escape it."""
    path = sample_dir / name
    if not path.resolve().is_relative_to(sample_dir.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="no such sample")
    return path


@app.get("/")
def index():
    # no-store: the browser otherwise serves a cached index.html after an edit,
    # which looks exactly like "my change did nothing" and wastes real time.
    return FileResponse(
        STATIC / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/info")
def info():
    """Model metadata. The UI reads its model list, vocabularies and samples
    from here rather than hardcoding them, so adding a checkpoint needs no
    frontend change."""
    return JSONResponse({
        "default": next(iter(_loaded)),
        "models": [{
            "key": s.key,
            "label": s.label,
            "blurb": s.blurb,
            "samples": discover_samples(s.sample_dir),
            **s.detector.info(),
        } for s in _loaded.values()],
    })


@app.get("/api/sample/{key}/{name}")
def sample(key: str, name: str):
    return FileResponse(_guarded(_spec(key).sample_dir, name))


@app.post("/api/detect")
async def detect(
    queries: str = Form(...),
    model: str = Form(...),
    conf: float = Form(0.25),
    image: UploadFile | None = File(None),
    sample_name: str = Form(""),
):
    """Run detection. Accepts either an uploaded file or a bundled sample name."""
    import cv2

    spec = _spec(model)
    query_list = [q.strip() for q in queries.split("\n") if q.strip()]
    if not query_list:
        raise HTTPException(status_code=400, detail="no queries given")

    if image is not None:
        raw = await image.read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise HTTPException(status_code=400, detail="could not decode that image")
    elif sample_name:
        arr = cv2.imread(str(_guarded(spec.sample_dir, sample_name)))
    else:
        raise HTTPException(status_code=400, detail="provide an image or a sample_name")

    dets = spec.detector.predict(arr, query_list, conf=conf)

    _, buf = cv2.imencode(".png", arr)
    return JSONResponse({
        "detections": [d.as_dict() for d in dets],
        "image_width": arr.shape[1],
        "image_height": arr.shape[0],
        # Echo the image back so the browser draws boxes over exactly the
        # pixels the model saw, rather than re-deriving them client-side.
        "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        "queries": query_list,
        "model": spec.key,
    })


def discover_samples(sample_dir: Path) -> list[str]:
    if not sample_dir.exists():
        return []
    return sorted(p.name for p in sample_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=[m.key for m in MODELS],
                    help="load a subset (default: every checkpoint that exists)")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    wanted = [m for m in MODELS if not args.only or m.key in args.only]
    for spec in wanted:
        if not spec.checkpoint.exists():
            # A missing checkpoint is normal mid-project (that model has not
            # finished training yet). Skip it rather than refusing to start,
            # so the app is usable with whatever is ready.
            print(f"skip {spec.key:<6} - not trained yet ({spec.run})")
            continue
        print(f"load {spec.key:<6} {spec.checkpoint} ...")
        spec.detector = DefectDetector(spec.checkpoint, device=args.device)
        _loaded[spec.key] = spec

    if not _loaded:
        raise SystemExit("no checkpoints found - train one first, or pass --only")

    print()
    for spec in _loaded.values():
        det = spec.detector
        warn = "" if det.clip_pretrained else "   WARNING: random text encoder!"
        print(f"  {spec.key:<6} {len(det.trained_vocabulary):>2} classes · "
              f"{len(discover_samples(spec.sample_dir)):>2} samples · {det.device}{warn}")
    print(f"\n  ->  http://{args.host}:{args.port}\n")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
