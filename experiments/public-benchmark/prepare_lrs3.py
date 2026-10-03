"""PROTOTYPE: unpack the mattymchen/lrs3-test parquet mirror into per-clip mouth arrays + refs.tsv."""
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[2] / "data/datasets/lrs3-test"
clips = root / "mouths"
clips.mkdir(exist_ok=True)
refs = []
for f in sorted(glob.glob(str(root / "raw/data/*.parquet"))):
    for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["idx", "label", "video"]):
        for row in batch.to_pylist():
            clip_id = f"{row['idx']:04d}"
            np.save(clips / f"{clip_id}.npy", np.array(row["video"], dtype=np.uint8))
            refs.append(f"{clip_id}\t{row['label'].strip()}")
(root / "refs.tsv").write_text("\n".join(sorted(refs)) + "\n")
print(len(refs), "clips")
