# Visual speech-activity detection (no audio)

Research for [#4](https://github.com/HyperToken9/lips/issues/4), part of map [#1](https://github.com/HyperToken9/lips/issues/1).

**Question:** with no audio, how do we tell when the single frontal Speaker is talking and when an utterance ends, so we can drop push to talk from the live webcam lip-reading loop (RTX 3060 Laptop, 6 GB)?

**Short answer:** no off-the-shelf visual-only VAD is both maintained and drop-in. Every strong active speaker detection (ASD) model needs audio. The best published visual-only numbers are about 89 to 94% clip accuracy on VVAD-LRS3, where humans score 88%. That is good enough for segmenting utterances once you add hysteresis and a hangover. Start with a **MediaPipe Face Landmarker mouth-motion heuristic**, which is cheap and Apache-2.0. If false triggers hurt, compare it against the **visual-only head already in Light-ASD's MIT checkpoint**.

---

## 1. What the evidence says about the ceiling

- **Visual-only is inherently weaker than audio-visual.** In the AVA-ActiveSpeaker paper, visual-only models reach 0.711 mAP (GRU, 2 stacked frames) against 0.821 for audio-visual. Adding audio cuts errors by more than 30 to 40%. The authors attribute the gap to "hard negatives that contain mouth motions", such as yawning or a hand at the mouth. [AVA-ActiveSpeaker, arXiv 1901.01342, Tables 4–7](https://arxiv.org/abs/1901.01342)
- **Face size matters.** Visual-only balanced accuracy is 74.5% on small faces and 84.2% on large faces (Table 6, same paper). A webcam Speaker close to the camera is in the easy "large face" regime.
- **Temporal context matters.** In VVAD-LRS3, one frame gives about 70 to 73% accuracy. Accuracy keeps rising up to the maximum of 38 frames (about 1.5 s at 25 fps). The authors note that "speaking cannot be inferred from a low number of frames". [VVAD-LRS3, arXiv 2109.13789, Fig. 4b, Table 4](https://arxiv.org/abs/2109.13789)
- **The labels define "not speaking" as pauses longer than 1.5 s.** VVAD-LRS3 builds its negatives from transcript gaps of 1.5 s or more. The benchmark therefore says nothing about detecting short inter-word pauses. Our end-of-utterance hangover will need its own tuning. (Same paper, dataset construction section.)
- **Moving lips are not always speech.** Smiling, chewing and yawning cause false positives. This is the documented failure mode of pure lip-motion methods, as discussed in the VVAD-LRS3 related-work section citing Bendris et al.

## 2. Options surveyed

| Option | Needs audio? | Accuracy (published) | Latency / compute | Availability | License |
|---|---|---|---|---|---|
| **A. Mouth-landmark heuristic** (MediaPipe Face Landmarker: 478 landmarks + 52 blendshapes) | No | None published for this exact recipe. Landmark-feature models on VVAD-LRS3 reach **89% test** (LSTM on dlib lip landmarks) | Face Landmarker has a `LIVE_STREAM` mode and runs on CPU. The solutions page gives no latency numbers, so we must measure | `pip install mediapipe`, maintained | Code Apache-2.0. Docs CC-BY-4.0. Model cards for FaceDetector, FaceMesh-V2 and Blendshape are linked from the guide |
| **B. Pretrained visual VAD: `vvadlrs3`** (Lubitz et al.) | No | 92% test (MobileNet + LSTM, face images), 89% (landmarks) | MobileNet, about 4.2M params, input of 38 frames at 96×96 | PyPI 0.2.0, last release **Apr 2021**, pins **TensorFlow 2.3.1 / Keras 2.4.3** plus dlib | Package LGPLv2. Its own note: the iBUG 300-W license for the landmark features **excludes commercial use**. Repo now GPL-3.0 |
| **B'. `sariyanidi/VoiceActivityDetection`** | No | Not stated | Needs the 3DI 3D face reconstruction for landmarks | 6 stars, last push Feb 2024 | NOASSERTION (unclear) |
| **C. Light-ASD** (CVPR 2023) | **Yes** for its headline result | 94.1% mAP on AVA (audio-visual). Visual-only head **not reported** | 1.0M params, 0.6 GFLOPs, 0.1 to 4.5 ms per frame on an RTX 3090. Input: 112×112 grayscale face crops at 25 fps | Weights in repo (`pretrain_AVA_CVPR.model`, `finetuning_TalkSet.model`, about 4 MB) | MIT |
| **C'. TalkNet-ASD** | **Yes** | 92.3 mAP val / 90.8 test on AVA (audio-visual) | ResNet-style visual front end. Demo uses S3FD face detector, 25 fps, 112×112 centre crop | Weights auto-downloaded. Last push Oct 2023 | MIT |
| **C''. LoCoNet** (CVPR 2024) | **Yes** | 95.2% mAP on AVA (audio-visual) | Heavier, multi-GPU training code | Weights on Google Drive | **No license file** (all rights reserved by default) |
| **D. Lip-reading encoder features** (for example auto_avsr VSR) | No | No published VAD result. Would need a probe trained on, say, VVAD-LRS3 | The front end already runs in our loop, so a linear probe is nearly free. But it is a 250M-param model that today runs per utterance, not per frame | auto_avsr checkpoints (Google Drive) | Code Apache-2.0. Checkpoints carry "their own licenses or terms … derived from the dataset" |

Sources: [MediaPipe Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker), [blendshape enum in `face_landmarker.py`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py) (`JAW_OPEN=25`, `MOUTH_CLOSE=27`, `MOUTH_FUNNEL`, `MOUTH_PUCKER`, `MOUTH_LOWER_DOWN_*`, `MOUTH_UPPER_UP_*` …), [vvadlrs3 on PyPI](https://pypi.org/project/vvadlrs3/), [adrianlubitz/VVAD](https://github.com/adrianlubitz/VVAD), [sariyanidi/VoiceActivityDetection](https://github.com/sariyanidi/VoiceActivityDetection), [Light-ASD repo](https://github.com/Junhua-Liao/Light-ASD) and [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liao_A_Light_Weight_Model_for_Active_Speaker_Detection_CVPR_2023_paper.html) (Tables 3 and 8), [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) and [`demoTalkNet.py`](https://github.com/TaoRuijie/TalkNet-ASD/blob/main/demoTalkNet.py), [LoCoNet_ASD](https://github.com/SJTUwxz/LoCoNet_ASD) and [arXiv 2301.08237](https://arxiv.org/abs/2301.08237), [auto_avsr README](https://github.com/mpc001/auto_avsr). Licenses were read from the GitHub license API.

### The visual-only head hidden in Light-ASD (and TalkNet)

Both repos train an auxiliary **visual-only** classifier next to the audio-visual one. In Light-ASD, `ASD.py` computes `outsV = self.model.forward_visual_backend(visualEmbed)` and adds `0.5 * lossV`. `lossV` holds its own `nn.Linear(128, 2)`. `saveParameters` stores the full `state_dict()`, so the released checkpoint contains the trained `lossV.FC` weights ([`ASD.py`](https://github.com/Junhua-Liao/Light-ASD/blob/main/ASD.py), [`loss.py`](https://github.com/Junhua-Liao/Light-ASD/blob/main/loss.py), [`model/Model.py`](https://github.com/Junhua-Liao/Light-ASD/blob/main/model/Model.py)).

That gives a pretrained, MIT-licensed, about 1M-param, audio-free speaking score per frame: `visualEncoder` → `lossV.FC` → softmax. Its accuracy has **never been reported**. It was only an auxiliary loss, and without audio it skips the BGRU, so expect it to fall well below the 94.1 audio-visual figure. It must be measured on our data before we trust it. TalkNet follows the same pattern (`0.4 * nlossV`).

## 3. Recommendation: what to experiment with

**Primary: mouth-motion heuristic on MediaPipe Face Landmarker** (option A). It is maintained, Apache-2.0, needs no GPU, and plugs into `LIVE_STREAM` mode. We probably need face landmarks for mouth cropping anyway. Sketch:

1. Per frame, compute a scale-normalised inner-lip aperture: the vertical distance between the inner lips divided by mouth width or inter-ocular distance. Also take the `jawOpen`, `mouthClose`, `mouthFunnel` and `mouthPucker` blendshapes.
2. Speech score = rolling standard deviation (or mean absolute first difference) of those signals over about 0.5 to 1 s. Use **motion, not openness**, so a mouth held open (yawning, smiling) is not counted as speech.
3. Add **hysteresis**: start an utterance when the score stays above `T_on` for about 150 to 250 ms. End it when the score stays below `T_off` for a **hangover** of about 0.6 to 1.0 s, then hand the buffered segment, with about 0.3 s of pre-roll, to the lip reader. These figures are starting points to tune, not sourced numbers.
4. Calibrate the thresholds per session from a few seconds of the Speaker sitting silently.

**Comparison: Light-ASD visual-only head** (option C). Run it on the same face track (112×112 grayscale, 25 fps) and compare frame-level and utterance-boundary agreement with the heuristic. It is the cheapest learned model that is already pretrained and has a clean license. If it handles smiling and chewing better, use it, or fuse it with the heuristic.

**Deprioritise:**
- `vvadlrs3`: dead TF 2.3 stack, LGPL/GPL, and a non-commercial clause on its landmark models.
- TalkNet, LoCoNet and audio-visual Light-ASD as designed: all need audio. LoCoNet also has no license.
- Probing the lip-reading encoder (D): worth revisiting only if the lip-reading model becomes streaming. Its features would come for free then, but it needs labelled training data and our current models run per utterance.

**How to judge it:** record a few minutes of the user talking with natural pauses, plus silent distractors (smiling, chewing, nodding), and hand-label speech spans. Report frame-level F1, the start and end boundary error in ms, false-trigger rate per minute while silent, and end-of-utterance latency (hangover plus processing). VVAD-LRS3 ([Kaggle](https://www.kaggle.com/datasets/adrianlubitz/vvadlrs3)) can serve as a secondary public check. Its clips are LRS3/TED-derived, so check the terms before any product use.

## 4. Open points

- No source publishes MediaPipe Face Landmarker latency on our hardware. It has to be measured.
- Accuracy of Light-ASD's visual-only head: unknown, measure it.
- The hangover length trades lag against cutting utterances at mid-sentence pauses. It interacts with the "utterance-level lag" goal in #1 and needs tuning on the user's own recordings.
