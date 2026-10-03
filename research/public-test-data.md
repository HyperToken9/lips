# Public lip-reading test data: what we can get, and on what terms

Research for [#3](https://github.com/HyperToken9/lips/issues/3) (part of map #1). Checked against primary sources on 2026-10-03.

## Answer

- **Primary test set: the LRS3 test split (1,321 utterances, about 1 hour).** The official download is gone, but the full test split is mirrored on Hugging Face. One mirror is ungated, 677 MB, and already preprocessed (`mattymchen/lrs3-test`). Every published model reports WER on this split, so it is the only set where we can check our numbers against the papers.
- **Secondary test set: WildVSR (2,854 utterances, 4.8 h, 88 MB zip, no sign-up).** It is harder and closer to our webcam case, and it was built to catch models that overfit LRS3.
- **Do not plan on LRS2, LRW or TCD-TIMIT.** BBC and TCD release them only to university or public-body researchers. An individual cannot get them.
- **Licensing:** the LRS3 annotations were CC BY-NC-ND 4.0 under the original terms, and the videos remain the property of their owners. Treat every set here as **evaluation-only, non-commercial**. Never put any of it in a product or a training set, and never push it to git (it goes in `data/`).

## Comparison

| Set | Content | Test size | How to get it (today) | Download | Terms | Fit to our case |
|---|---|---|---|---|---|---|
| **LRS3-TED** | TED/TEDx talks, open vocabulary | 1,321 utt, 412 videos, ~1 h, 2k vocab | Official: **gone**. HF mirrors (see below) | Test only: 677 MB (`mattymchen/lrs3-test`) | Original annotations: CC BY-NC-ND 4.0. Later KAIST page: CC BY 4.0, copyright with video owners | **Best.** One mostly frontal speaker, prepared sentences, open vocabulary |
| **WildVSR** | Creative Commons YouTube (interviews, talks, etc.), built like LRS3 | 2,854 utt, 618 speakers, 4.8 h, 6k vocab | Google Drive link in the GitHub repo, no sign-up | 88 MB zip (cropped clips + `labels.json`) | Repo has no license file. Collected from CC-licensed YouTube; research/evaluation use stated | **Good, harder.** Single speaker, open vocabulary, more head motion and low resolution |
| **LRS-VoxMM** (2026) | VoxMM conversations, 12 domains | Test 2,146 utt / 1.8 h, dev 27k utt / 23.5 h | Key request form at KAIST (`cn01.mmai.io/keyreq/voxmm`) | not stated | CC BY 4.0 "for research purposes", copyright with video owners | Fair. Conversational, in the wild, much harder (VSR WER 55–71%) |
| **LRS2-BBC** | BBC TV | 1,243 utt, ~0.5 h | BBC agreement. **Institutions only**, so not open to us | 50 GB package | Non-commercial academic only, 12-month permission, no redistribution | Good fit, but **we cannot get it** |
| **LRW** | BBC TV, isolated words | 25k clips, 500 words | Same BBC agreement, so not open to us | 70 GB | Same as LRS2 | Poor. Word classification, not sentences |
| **GRID** | Lab recordings, 34 speakers | 34k sentences (no standard sentence split) | Zenodo, open | 16.2 GB | **CC BY 4.0** | Poor. Fixed grammar ("put red at G9 now"), 51-word vocabulary. Only useful as a sanity check |
| **TCD-TIMIT** | Lab, 62 speakers (3 lipspeakers), frontal + 30° | 6,913 TIMIT sentences | Account at sigmedia.tcd.ie, **university email required** | not stated | Non-commercial only | Decent (frontal, read sentences), but **not open to us** |
| **VoxCeleb1/2, AVSpeech** | YouTube faces | none | — | — | — | **No transcripts.** Used for pretraining, not usable as a WER test set |

## Details and sources

### LRS3-TED: is the test split obtainable? Yes, through mirrors

- **Official status.** The KAIST/mmai page says: "Downloads are no longer available from this website. However, we have recently released LRS-VoxMM as a new benchmark for lip-reading research." ([mm.kaist.ac.kr/datasets/lip_reading](https://mm.kaist.ac.kr/datasets/lip_reading/)). The Oxford VGG page `robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html` now returns 404 (the Wayback Machine shows it was already 404 by January 2025).
- **Original terms** (Wayback snapshot, 2023-01-08, [VGG LRS3 page](http://web.archive.org/web/20230108172440/https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html)): "The annotations are licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License." Video files required a password requested through a form, and "Password issued for LRW and LRS2 datasets can also be used to download LRS3." Test split v0.4: "412 folders and 1,321 utterances."
- **Later terms.** The KAIST page states CC BY 4.0, "with copyright retained by original video owners." The two statements conflict. Since TED content itself is non-commercial, assume the stricter NC-ND reading.
- **Mirror A: `mattymchen/lrs3-test`** ([HF](https://huggingface.co/datasets/mattymchen/lrs3-test)). Ungated. Two parquet files, 677 MB in total, 1,321 rows. I checked one shard myself. Each row has `video` as a uint8 array of shape T×96×96 (grayscale mouth crop at 25 fps), `audio` as int16 at 16 kHz, and `label` as a lowercase transcript (e.g. "the first lesson is about humility"). That is the 96×96 mouth-ROI format the Auto-AVSR and AV-HuBERT pipelines use. **Caveat:** these are pre-cropped mouths, not raw face video, so a model whose own cropping pipeline differs (landmarks and alignment) may score slightly differently than on the raw clips. The card does not state a license.
- **Mirror B: `TheNHz/ellipsis-lrs3-raw`** ([HF](https://huggingface.co/datasets/TheNHz/ellipsis-lrs3-raw)). Gated with automatic approval: you log in to HF and agree to share contact info. It is a "verified mirror" of the full official dataset (150 GB). Its test split is "1,321 utterances, complete", but the files under `test-mattymchen/` are the same two parquet files as Mirror A (identical blob IDs). It claims CC BY 4.0 and reports that Prof. Zisserman confirmed by email that the official distribution is discontinued (I have not verified that email). Pretrain is 87.3% complete. This is the place to get the 224×224 face-track clips if raw faces are ever needed.
- **Fallback:** if the mirrors disappear, WildVSR (below) is the open replacement. The LRS3 annotation files list YouTube IDs and frame ranges, so the clips can in principle be rebuilt from YouTube, but that is slow and lossy because videos get removed.

### WildVSR

- Repo: [YasserdahouML/VSR_test_set](https://github.com/YasserdahouML/VSR_test_set). Paper: [Do VSR Models Generalize Beyond LRS3? (arXiv 2311.14063)](https://arxiv.org/abs/2311.14063).
- Download: a public Google Drive file, `WildVSR.zip`, **88 MB**. The layout is `videos/*.mp4` plus `labels.json`. "All clips are cropped and transformed."
- Stats from the paper: 4.8 h, 2,854 utterances, 618 speakers, 6,040 vocabulary, 478 Creative Commons YouTube videos, all manually verified.
- Published VSR WER, LRS3 → WildVSR: Auto-AVSR 19.1 → 38.6, AV-HuBERT Large 28.6 → 48.7, RAVen Large 23.1 → 46.7. The paper fits WER(WildVSR) ≈ 1.31 × WER(LRS3) + 14.05.
- License: the repo has no LICENSE file. The README says the set uses "only free-to-use content" and gives the intended use as testing VSR models for research. Treat it as evaluation-only.

### LRS-VoxMM (newest, 2026)

- Paper: [arXiv 2604.27866](https://arxiv.org/abs/2604.27866) (submitted 2026-04-30). Project page: [mm.kaist.ac.kr/projects/voxmm](https://mm.kaist.ac.kr/projects/voxmm/). Preprocessing code: [kaistmm/VoxMM](https://github.com/kaistmm/VoxMM).
- Evaluation only: dev has ~698 speakers, 27k utterances, 23.5 h; test has ~113 speakers, 2,146 utterances, 1.8 h. It is in LRS format, with transcripts normalized to LRS2/3 conventions.
- VSR WER (dev/test): AV-HuBERT 59.7/65.8, Auto-AVSR 47.4/55.2, Llama-AVSR 62.9/70.7. On LRS3 the same models score 20.6–27.2.
- Access: a key request form at `cn01.mmai.io/keyreq/voxmm`, which refused connections when I checked, so I could not confirm the eligibility rules. The license is "for research purposes under CC BY 4.0", with copyright staying with the video owners.
- Fit: conversational, multi-domain, more non-frontal views. It is harder than our case. Optional, as a stress test.

### LRS2 and LRW (BBC): closed to individuals

From the [BBC R&D datasets page](https://www.bbc.co.uk/rd/projects/lip-reading-datasets):
- "The datasets are available to researchers from universities and other reputable academic institutions and relevant public organisations, strictly for non-commercial research. Use is not permitted by commercial organisations."
- "You must use your official academic email address … Gmail or non academic email accounts are not acceptable."
- "**Use is not permitted by companies or independent researchers.**"
- Permission lasts 12 months. No redistribution. Forms must be sent as Word `.doc` files.

Sizes are from the VGG pages: LRS2 is 50 GB (test 1,243 utterances), [lrs2.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html); LRW is 70 GB, [lrw1.html](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html).

### GRID

[Zenodo record 3625687](https://zenodo.org/records/3625687): CC BY 4.0, open, 16.2 GB, 34 talkers × 1,000 sentences, video for 33 speakers. The fixed six-word grammar makes WER on it meaningless for open-vocabulary models. Skip it, or use a handful of clips as a smoke test.

### TCD-TIMIT

[sigmedia.tcd.ie](https://sigmedia.tcd.ie/): "To access the data, you need to create an account. Your account will only be enabled if you supply a valid university email address and properly identify yourself." The license is non-commercial only ([sigmedia.tv](https://sigmedia.tv/datasets/tcd_timit/)). It has 62 speakers and 6,913 sentences, recorded frontal and at 30°. It would fit well, but an individual cannot get it.

### VoxCeleb / AVSpeech

These are speaker-ID and separation corpora with no transcripts, so they cannot be used for WER. Models use them only for pretraining (e.g. AV-HuBERT is pretrained on VoxCeleb2, per the LRS-VoxMM paper).

## Recommendation

1. **Benchmark every candidate model on the LRS3 test split from `mattymchen/lrs3-test`** (677 MB, no sign-up). Use it to check that our WER matches each model's published LRS3 number. If it does not, our pipeline is wrong.
2. **Report WildVSR alongside it** (88 MB, no sign-up), as the realistic generalization number.
3. If a model needs raw face video instead of 96×96 mouth crops, get the LRS3 test split from `TheNHz/ellipsis-lrs3-raw` (HF login plus click-through) and run that model's own cropping.
4. LRS-VoxMM is optional. Request a key only if we want a conversational stress test.
5. Skip LRS2, LRW and TCD-TIMIT (not available to an individual) and GRID (fixed grammar).

**Access barriers for an individual:** none for steps 1–2. Step 3 needs a free Hugging Face account and a click-through on the gate. BBC and TCD explicitly exclude independent researchers.

**Licensing note for #1:** all of these sets are research-only or of unclear provenance (LRS3 annotations NC-ND, WildVSR has no license, the mirrors are unofficial re-hosts). Use them only to measure WER. Keep them in git-ignored `data/` and never in a shipped product or training set.
