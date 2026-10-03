"""PROTOTYPE: unpack WildVSR's 96x96 mouth mp4s into per-clip grayscale arrays + refs.tsv."""
import json
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[2] / "data/datasets/wildvsr"
src = root / "WildVSR"
out = root / "mouths"
out.mkdir(exist_ok=True)
labels = json.loads((src / "labels.json").read_text())
refs = []
for name, text in sorted(labels.items()):
    cap = cv2.VideoCapture(str(src / "videos" / name))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    clip_id = Path(name).stem
    np.save(out / f"{clip_id}.npy", np.stack(frames))
    refs.append(f"{clip_id}\t{text.strip()}")
(root / "refs.tsv").write_text("\n".join(refs) + "\n")
print(len(refs), "clips")
