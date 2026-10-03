# Public benchmark: which model reads the public test sets best?

Ticket: [Which model reads the public test set best?](https://github.com/HyperToken9/lips/issues/6). **PROTOTYPE** code: throwaway, kept for reproducibility.

## Verdict

**Auto-AVSR** (`vsr_trlrs2lrs3vox2avsp_base.pth`, Apache-2.0 code from `mpc001/auto_avsr`) is the model the live prototype builds on.

- Use **greedy decoding** (beam 1, no CTC) for live use: 38.1% WER on WildVSR at 0.032× real time, 1.4 GB VRAM. Three seconds of speech decodes in about 0.1 s.
- Use **beam 40** when latency doesn't matter: 35.8% WER at 0.32× real time, 1.8 GB.

It beats both USR 2.0 checkpoints on accuracy, speed and memory on this 6 GB GPU.

## Results

Hardware: RTX 3060 Laptop (6 GB), PyTorch 2.5.1 + CUDA 12.4. Speed is compute time ÷ video duration, so lower is faster and below 1 keeps up with real time. Out-of-memory (OOM) clips are scored as empty Transcripts.

**WildVSR** (first 600 of 2,854 clips, 60 min of video; this is the deciding set):

| Model | Decoding | WER | CER | Speed | Peak VRAM | OOM clips |
|---|---|---|---|---|---|---|
| **Auto-AVSR** | beam 40 | **35.8%** | 24.5% | 0.32 | 1.8 GB | 0 |
| **Auto-AVSR** | greedy | 38.1% | 25.7% | **0.032** | 1.4 GB | 0 |
| USR 2.0 Huge (fp16 encoder) | greedy | 40.4% | 27.7% | 0.075 | 2.7 GB | 0 |
| USR 2.0 Base+ | greedy | 55.2% | 37.8% | 0.034 | 1.1 GB | 0 |
| USR 2.0 Base+ | beam 10 | 57.2% | 41.9% | 0.68 | 5.3 GB | 22 |

Published WildVSR WER: Auto-AVSR 38.6% (with LM), USR 2.0 Huge 38.5%. Our Auto-AVSR number matches, which validates the pipeline.

**LRS3 test** (all 1,321 clips, 50 min, from the `mattymchen/lrs3-test` mirror; secondary, see the caveat below):

| Model | Decoding | WER | CER | Speed | Peak VRAM |
|---|---|---|---|---|---|
| Auto-AVSR | beam 40 | 26.8% | 18.1% | 0.32 | 1.3 GB |
| Auto-AVSR | greedy | 29.8% | 19.9% | 0.034 | 1.2 GB |
| USR 2.0 Base+ | beam 10 | 31.7% | 21.1% | 0.42 | 2.1 GB |
| USR 2.0 Base+ | greedy | 35.7% | 23.6% | 0.039 | 0.9 GB |

## Findings

1. **The LRS3 mirror's crops are misaligned.**
   - Every model scores 6–7 points worse than published (Auto-AVSR: 26.8% vs 20.3%), with errors spread evenly across the set.
   - Its crops include rotated, profile and off-centre mouths, while WildVSR's are consistently aligned. See `data/runs/public-benchmark/inspect/crops.png`: the top row is LRS3, the bottom WildVSR.
   - On WildVSR the same pipeline matches published numbers. So LRS3 from this mirror is only good for relative comparisons.
   - This also shows that **face and mouth alignment quality matters a lot**, which the live webcam pipeline has to get right.
2. **Beam search is a memory problem on 6 GB.**
   - USR 2.0's default beam 40 runs out of memory on 6 s clips even for Base+.
   - Huge (2.3 GB of weights with an fp16 encoder, 3.6 GB in fp32) runs out of memory at beam 10 on long WildVSR clips.
   - fp16 autocast doesn't help, because the memory goes to per-hypothesis decoder state.
   - Auto-AVSR's beam search stays under 2 GB.
3. **Greedy decoding costs only 2–4 WER points but is 10–20× faster.** All greedy setups fit easily, so greedy is the natural live mode.
4. **USR 2.0 underdelivers here.**
   - Huge greedy (40.4%) is behind Auto-AVSR greedy (38.1%), and its beam search, where its published 38.5% comes from, doesn't fit this GPU on long clips.
   - Base+ generalises poorly from LRS3 (31.7%) to WildVSR (55–57%).
5. **USR 2.0 Huge needs fp16 for its encoder only.** The CTC prefix scorer fills with -1e10, which overflows fp16, so the decoder and CTC stay fp32.

## Licenses

- **Auto-AVSR:** code Apache-2.0 (`mpc001/auto_avsr`). Weights trained on LRS2, LRS3, VoxCeleb2 and AVSpeech, which are non-commercial.
- **USR 2.0:** code MIT (`ahaliassos/usr2`). Weights trained on the same non-commercial data.
- **Datasets:** the LRS3 mirror (CC BY-NC-ND annotations, unofficial re-host) and WildVSR (no license file) are evaluation-only.

**All weights are research-only.** A product would need weights trained on commercially licensed data.

## Reproduce

```sh
cd experiments/public-benchmark
uv sync                                   # torch 2.5.1+cu124 etc.; prefix with PYTHONPATH= on this machine
git clone https://github.com/mpc001/auto_avsr ../../data/vendor/auto_avsr
git clone https://github.com/ahaliassos/usr2  ../../data/vendor/usr2
# checkpoints → ../../data/checkpoints/ (names in bench.py MODELS; Google Drive links in each repo's README)
uv run python prepare_lrs3.py             # after: hf download mattymchen/lrs3-test → data/datasets/lrs3-test/raw
uv run python prepare_wildvsr.py          # after: unzip WildVSR.zip → data/datasets/wildvsr/WildVSR
./queue.sh "autoavsr wild-autoavsr-greedy --beam 1 --ctc-weight 0 --dataset ../../data/datasets/wildvsr --limit 600"
```

- `bench.py`: runs one model over a folder of 96×96 mouth crops. It writes `hyps.tsv`, `timing.tsv` and `summary.json`, resumes after a crash, and records OOM clips.
- `queue.sh`: runs jobs strictly one at a time (two models don't fit on the GPU together) and scores each with `lips-wer`.
- Run outputs live in `data/runs/public-benchmark/<run>/`.
