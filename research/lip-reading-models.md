# Which pretrained lip-reading models are viable on a 6 GB laptop GPU?

Research for [#2](https://github.com/HyperToken9/lips/issues/2) (part of map [#1](https://github.com/HyperToken9/lips/issues/1)). Researched 2026-10-03.
Target hardware: RTX 3060 Laptop (6 GB VRAM), 13 GB RAM, Python 3.10.

## Answer

The ResNet-frontend + Transformer/Conformer encoder-decoder family from the Imperial/Meta group (Auto-AVSR, USR, USR 2.0) is the practical sweet spot. Checkpoints are public, VSR-only inference runs on short clips, the models are under about 1 B params (all fit in 6 GB in fp16, most in fp32 too), they include a mouth-crop pipeline with a MediaPipe option, and the code is MIT/Apache.
The LLM-based systems (Llama-AVSR, VSP-LLM, VALLR) either have no released weights or need a 7–8 B LLM, and they don't beat USR 2.0 on WER.

**Shortlist to benchmark:**

1. **USR 2.0 Huge (high-resource)**: best published open checkpoint, **17.6 % LRS3 / 12.6 % LRS2** VSR WER, 953 M params, MIT code, ships a `demo.py` with MediaPipe cropping.
2. **USR 2.0 Base+ (high-resource)**: the light option. **24.8 % LRS3**, 171 M params, same code path. This is the latency baseline for live use.
3. **Auto-AVSR VSR (mpc001)**: **19.1 % LRS3** (with LM; 20.3 % without, in `auto_avsr`), about 250 M params. Mature `infer.py` with RetinaFace/MediaPipe and a Colab tutorial. An independent second family to check USR 2.0 against.
4. *(Optional)* **AV-HuBERT Large VSR** (26.9 % LRS3 with self-training) is a reference point only. It's on a non-commercial Meta license, its archived fairseq stack expects Python 3.8, and it scores worse than 1–3.

**Licensing caveat for all of them:** every checkpoint is trained on LRS3 (CC BY-NC-ND 4.0), and most also on VoxCeleb2 (CC BY-NC-ND 4.0) and/or LRS2 (BBC, non-commercial research only). Permissive *code* licenses don't make the *weights* commercially usable. Treat every candidate as research-only until a lawyer says otherwise.

## Candidate table

WER is VSR (video-only) on the LRS3 test set unless noted. Lower is better.

| Model | LRS3 WER | LRS2 WER | Params | Checkpoint | Code license | Weights / data terms | 6 GB feasible? |
|---|---|---|---|---|---|---|---|
| **USR 2.0 Huge** (high-res) | **17.6** | **12.6** | 953 M | Google Drive | MIT | Trained on LRS2+LRS3+Vox2+AVSpeech (NC) | Yes in fp16 (~1.9 GB of weights). Unverified, measure |
| USR 2.0 Large (high-res) | 21.5 | – | 503 M | Google Drive | MIT | LRS3+Vox2 (NC) | Yes |
| **USR 2.0 Base+** (high-res) | **24.8** | – | 171 M | Google Drive | MIT | LRS3+Vox2 (NC) | Yes, comfortably |
| USR (v1) Large (high-res) | 22.3 | – | ~503 M | Google Drive | **no LICENSE file** | LRS3+Vox2 (NC) | Yes. Superseded by USR 2.0 |
| **Auto-AVSR VSR** (3,448 h) | **19.1** (LM) / 20.3 | 14.6 (paper, no public LRS2 ckpt found) | ~250 M | Google Drive / Baidu | Apache-2.0 (`auto_avsr`); NC/benchmark-only (`Visual_Speech_Recognition_for_Multiple_Languages`) | LRS2+LRS3+Vox2+AVSpeech (NC) | Yes |
| VSR-for-Multiple-Languages (2022) | 32.3 | 26.1 | 186 MB file | Google Drive / Baidu | Custom BSD-like, **benchmark/non-commercial only** | LRS2/LRS3 (NC) | Yes. Obsolete vs Auto-AVSR |
| BRAVEn Large w/ ST+LM | 20.1 | – | Large | Yes (raven repo) | MIT | LRS3+Vox2+AVS (NC) | Yes. Inference scripts only, preprocessing heavier |
| RAVEn Large w/ ST+LM | 23.1 | – | Large | Yes | MIT | LRS3+Vox2 (NC) | Yes. Superseded by BRAVEn/USR |
| AV-HuBERT Large + ST | 26.9 | – | ~325 M | Yes (Meta CDN) | **AV-HuBERT License: non-commercial research only**, bans surveillance/biometric use | LRS3+Vox2 (NC) | Yes, but fairseq fork / Py 3.8; repo archived |
| Llama-AVSR VSR | 23.7 | – | AV-HuBERT-L + Llama-2-7B (LoRA) | Yes | no SPDX license detected | Llama 2 license + AV-HuBERT license | Only with 4-bit LLM; marginal, not worth it |
| VSP-LLM | 25.4 | – | AV-HuBERT + LLaMA2-7B (QLoRA) | Code yes | NOASSERTION | Llama 2 + AV-HuBERT | Marginal, as above |
| VALLR (ICCV 2025) | 18.7 | 20.8 | ViT-B + Llama-3.2-3B | **Not released** ("after review") | – | – | n/a |
| torchaudio real-time AV-ASR (Emformer RNN-T) | VSR not reported | – | 35 M / 383 M | **Recipe only, no VSR checkpoint** | BSD-2 | – | n/a (would need training) |

## Per-model notes

### USR 2.0: `ahaliassos/usr2` (ICLR 2026)
- Paper: *Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition*, Haliassos, Mira, Petridis. [arXiv:2602.19316](https://arxiv.org/abs/2602.19316). Code: <https://github.com/ahaliassos/usr2> (created 2026-01, last push 2026-09, MIT `LICENSE`).
- One unified model for ASR, VSR and AVSR. Run `modality=v` for lip reading only.
- Fine-tuned checkpoints (README "Pretrained Models"). LRS3 VSR WER: low-resource Base 36.2 / Base+ 26.4 / Large 23.7. High-resource Base+ 24.8 / Large 21.5 / **Huge 17.6**. Huge is trained on LRS2+LRS3 labelled data plus English VoxCeleb2+AVSpeech unlabelled. The paper reports Huge at 12.6 % on LRS2 and 38.5 % on WildVSR, no external LM (paper Tables 7–9).
- Params (paper Table 5, incl. both frontends + decoder): Base 86 M, Base+ 171 M, Large 503 M, Huge 953 M.
- Preprocessing (paper A.3): stabilise, crop 96×96 around the mouth, grayscale. `demo.py` does face detection and mouth cropping automatically. **MediaPipe is the default (CPU)**. RetinaFace+FAN (ibug packages, CUDA) is optional, with `detector=retinaface`.
- Decoding: joint CTC/attention beam search, default beam 40, CTC weight 0.1. The README's speed tip is `decode.beam_size=1 decode.ctc_weight=0.0` for greedy. The paper notes that CTC-only decoding is ~40× faster than autoregressive decoding.
- Speed and VRAM: not published. The weight sizes suggest everything fits in 6 GB. **Measure** latency per utterance.
- Streaming: the encoder is full-context (non-causal), so live use would be utterance-level (push-to-talk segment, then decode). That matches the map's "utterance-level lag" goal. True word-by-word streaming would need retraining (out of scope).

### USR (v1): `ahaliassos/usr` (NeurIPS 2024)
- [arXiv:2411.02256](https://arxiv.org/abs/2411.02256). Checkpoints: high-resource Base 34.3 / Base+ 26.5 / Large 22.3 LRS3 VSR. Preprocessing uses RetinaFace + 2-D FAN.
- **The repo has no LICENSE file** (GitHub API `license: null`), so by default all rights are reserved. USR 2.0 supersedes it on both WER and licensing. Skip it.

### Auto-AVSR: `mpc001/auto_avsr` and `mpc001/Visual_Speech_Recognition_for_Multiple_Languages`
- Paper: Ma et al., *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels*, ICASSP 2023 ([arXiv:2303.14307](https://arxiv.org/abs/2303.14307)).
- `auto_avsr` model zoo (Apache-2.0 code): `vsr_trlrs2lrs3vox2avsp_base.pth`, 3,291 h, **20.3 %** LRS3, 250 M params. The README says: "The pre-trained models provided in this repository may have their own licenses or terms and conditions derived from the dataset used for training."
- `Visual_Speech_Recognition_for_Multiple_Languages` model zoo: the Auto-AVSR visual-only LRS3 model is **19.1 %** (891 MB file), with a separate 191 MB LM. It has an inference CLI, `python infer.py config_filename=... data_filename=... [detector=mediapipe]`, RetinaFace by default, and MediaPipe switchable. It also has a Colab tutorial and runs on CPU with `gpu_idx=-1`.
- **License trap:** that second repo's LICENSE adds clauses saying the content can "only be used for comparative or benchmarking purposes", and its README says "Users can only use code supplied under a License for non-commercial purposes." That's fine for this research, but prefer `auto_avsr` (Apache) code paths if anything gets reused.
- Auto-AVSR paper LRS2 VSR WER is 14.6 % (3,448 h, as cited in USR 2.0 Table 8). No public LRS2-specific Auto-AVSR checkpoint was found.
- Streaming: same as USR (offline conformer encoder-decoder), so utterance-level only.

### BRAVEn / RAVEn: `ahaliassos/raven` (MIT)
- BRAVEn Large w/ self-training + LM: 20.0 % (low-resource) / 20.1 % (high-resource) LRS3. RAVEn best is 23.1 %.
- The repo only has inference scripts. Pre-training/fine-tuning code was "coming soon". Preprocessing needs RetinaFace + 2-D FAN landmarks. These are the same authors as USR, and USR 2.0 beats them, so there's no reason to benchmark separately.

### AV-HuBERT: `facebookresearch/av_hubert`
- Paper: Shi et al., ICLR 2022 ([arXiv:2201.02184](https://arxiv.org/abs/2201.02184)). The abstract gives 32.5 % WER with 30 h labelled data and **26.9 %** with 433 h + self-training (Large, LRS3+Vox2 pre-training). Base+/433 h is 34.8 % and Large without self-training is 28.6 % (as tabulated in the USR 2.0 paper).
- VSR fine-tuned checkpoints for Base/Large and 30 h/433 h, with and without self-training, are on <https://facebookresearch.github.io/av_hubert/>.
- **License**: the "AV-HuBERT LICENSE AGREEMENT" (Meta) grants rights "solely for your non-commercial research purposes" and forbids use "for (i) any commercial or production purposes … (iii) purposes of surveillance … (iv) biometric processing". It covers "any data produced by the Software".
- Practicalities: the repo is **archived** (last push 2023-12). It needs a pinned fairseq fork and the README creates a `python=3.8` conda env, which is a poor fit for Python 3.10 + `uv`. Preprocessing uses dlib CNN face detector + dlib 68 landmarks, then a 96×96 mouth crop.
- It matters mostly as the encoder inside the LLM systems below.

### LLM-based systems (2024–2026)
- **Llama-AVSR** (`umbertocappellazzo/Llama-AVSR`, ICASSP 2025/2026): the VSR checkpoint is AV-HuBERT Large + Llama-2-7B with LoRA, 23.7 % LRS3. It needs gated Llama weights. fp16 7 B ≈ 13 GB, so it won't fit in 6 GB without 4-bit quantisation, and it inherits the AV-HuBERT and Llama licenses. Not worth it.
- **VSP-LLM** (`Sally-SH/VSP-LLM`, EMNLP 2024 Findings): AV-HuBERT + LLaMA2-7B (QLoRA), 25.4 % LRS3. Same objections.
- **VALLR** (ICCV 2025, [arXiv:2503.21408](https://arxiv.org/abs/2503.21408)): ViT-B phoneme CTC + Llama-3.2-3B. Claims 18.7 % LRS3 / 20.8 % LRS2 from only 30 h labelled data. The paper says "Code will be released following the review process", and no checkpoint has been found. Watch it, but it can't be benchmarked.
- *From Hype to Insight* ([arXiv:2509.14880](https://arxiv.org/abs/2509.14880)) argues that much of the LLM gain in VSR is language modelling rather than better visual recognition. That's another reason to start with the compact models and add LLM post-correction separately if needed (the map already has a "correction" stage).

### torchaudio real-time AV-ASR (Emformer RNN-T)
- [PyTorch blog](https://pytorch.org/blog/real-time-speech-rec/), recipe in `pytorch/audio/examples/avsr` (BSD-2). It's the only genuinely **streaming** architecture here: Emformer transducer, RTF 0.33 on a laptop RTX 3070 Ti, face crop via Ultra-Light-Fast face detector.
- Results are reported for AV and audio only. **No pretrained VSR checkpoint is released**, only a training recipe. The project rules out training from scratch, so it's not viable now. It's a useful design reference for the "word-by-word streaming" follow-up.

## Preprocessing summary

All shortlisted models expect the same input: 25 fps video, face landmarks, a stabilised/aligned **96×96 grayscale mouth crop** (USR 2.0 A.3; Auto-AVSR uses the same pipeline). Both USR 2.0 and the mpc001 repo offer **MediaPipe** (CPU, light, good for a webcam loop) or **RetinaFace + FAN** (GPU, more accurate, adds VRAM). A webcam pipeline should resample to 25 fps.

## Hardware feasibility (estimates, to be measured)

- Weights only, fp16: Base+ ≈ 0.35 GB, Auto-AVSR ≈ 0.5 GB, Large ≈ 1 GB, Huge ≈ 1.9 GB. Activations for a ≤10 s utterance (≤250 frames at 96×96) are small. Beam size dominates decode time.
- No source publishes VRAM or latency for these checkpoints on a 3060-class GPU, so **this is the first thing the benchmark ticket should measure**: peak VRAM, encoder time, and decode time at beam 1/10/40.
- 13 GB system RAM is enough. Avoid loading several large checkpoints at once.

## Dataset note for the benchmark ticket

- LRS3 is listed as CC BY-NC-ND 4.0 in the USR 2.0 paper (A.1). LRS2 is "academic, non-commercial research use" only, and VoxCeleb2 is CC BY-NC-ND 4.0.
- A Hugging Face mirror card ([TheNHz/ellipsis-lrs3-raw](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw)) states that "As of 2026-07, the official downloads are gone" and claims a complete LRS3 test set (1,321 utterances). It labels LRS3 "CC BY 4.0", which **conflicts** with the paper's CC BY-NC-ND 4.0. I haven't verified this against the official page. Resolve it when the benchmark data is acquired.

## Sources

- USR 2.0 code and model zoo: <https://github.com/ahaliassos/usr2> (README, LICENSE). Paper: <https://arxiv.org/abs/2602.19316> (HTML v1: Tables 5, 7–9, Appendix A.1, A.3)
- USR: <https://github.com/ahaliassos/usr>, <https://arxiv.org/abs/2411.02256>
- Auto-AVSR: <https://github.com/mpc001/auto_avsr> (README model zoo, License section), <https://arxiv.org/abs/2303.14307>
- VSR for Multiple Languages: <https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages> (README model zoo, LICENSE)
- RAVEn/BRAVEn: <https://github.com/ahaliassos/raven>
- AV-HuBERT: <https://github.com/facebookresearch/av_hubert> (LICENSE, README, `avhubert/preparation/README.md`), checkpoints <https://facebookresearch.github.io/av_hubert/>, <https://arxiv.org/abs/2201.02184>
- Llama-AVSR: <https://github.com/umbertocappellazzo/Llama-AVSR>
- VSP-LLM: <https://github.com/Sally-SH/VSP-LLM>, <https://arxiv.org/abs/2402.15151>
- VALLR: <https://arxiv.org/abs/2503.21408>
- From Hype to Insight: <https://arxiv.org/abs/2509.14880>
- torchaudio real-time AV-ASR: <https://pytorch.org/blog/real-time-speech-rec/>, <https://github.com/pytorch/audio/tree/main/examples/avsr>
