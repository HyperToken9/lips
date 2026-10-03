"""PROTOTYPE: run one pretrained lip-reading model over a folder of 96x96 mouth crops.

Writes hyps.tsv (clip_id<TAB>text), timing.tsv and summary.json into --out, then
score with `uv run lips-wer`. One model per process: USR 2.0 and Auto-AVSR both
ship a top-level `espnet` package, so they can't share an interpreter.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "data/vendor"
CKPT = ROOT / "data/checkpoints"

MODELS = {
    "usr2-baseplus": ("usr2", "resnet_transformer_baseplus", "usr2_baseplus_high.pth"),
    "usr2-large": ("usr2", "resnet_transformer_large", "usr2_large_high.pth"),
    "usr2-huge": ("usr2", "resnet_transformer_huge", "usr2_huge_high.pth"),
    "autoavsr": ("autoavsr", None, "autoavsr_vsr_trlrs2lrs3vox2avsp_base.pth"),
}

# Shared by both families: scale, centre-crop the 96x96 mouth to 88x88, normalise.
TRANSFORM = torch.nn.Sequential(
    torchvision.transforms.CenterCrop(88),
    torchvision.transforms.Normalize(0.421, 0.165),
)


def load_usr2(backbone, ckpt, beam, ctc_weight, device, dtype):
    sys.path.insert(0, str(VENDOR / "usr2"))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from espnet.asr.asr_utils import parse_hypothesis
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.nets.scorers.length_bonus import LengthBonus
    from utils.utils import UNIGRAM1000_LIST as tokens

    OmegaConf.register_new_resolver("len", len, replace=True)
    with initialize_config_dir(str(VENDOR / "usr2/conf"), version_base="1.3"):
        cfg = compose("config", overrides=[f"model/backbone={backbone}"])

    model = E2E(len(tokens), cfg.model.backbone)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.backbone.") for k in state):
        state = {k.replace("model.backbone.", "", 1): v for k, v in state.items() if k.startswith("model.backbone.")}
    model.load_state_dict(state)
    # Only the encoder goes to `dtype`: the CTC prefix scorer fills with -1e10, which overflows fp16.
    model.eval().to(device)
    model.encoder.to(dtype)

    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(tokens))
    search = BatchBeamSearch(
        beam_size=beam, vocab_size=len(tokens),
        weights=dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, length_bonus=0.0),
        scorers=scorers, sos=len(tokens) - 1, eos=len(tokens) - 1, token_list=tokens,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    ).to(device)

    def run(video):  # video: (T, 88, 88) normalised
        feat = model.encoder(xs_v=video.unsqueeze(0)).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_enc = time.perf_counter()
        hyps = search(x=feat.squeeze(0), modality="v", maxlenratio=1.0, minlenratio=0.0)
        text, _, _, _ = parse_hypothesis(hyps[0].asdict(), tokens)
        return text.replace("<eos>", "").replace("▁", " ").strip(), t_enc

    return run


def load_autoavsr(ckpt, beam, ctc_weight, device, dtype):
    sys.path.insert(0, str(VENDOR / "auto_avsr"))
    from datamodule.transforms import TextTransform
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from espnet.nets.scorers.length_bonus import LengthBonus

    text_transform = TextTransform()
    tokens = text_transform.token_list
    model = E2E(len(tokens), "video", ctc_weight=ctc_weight)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(device)
    for part in (model.frontend, model.proj_encoder, model.encoder):
        part.to(dtype)

    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(tokens))
    search = BatchBeamSearch(
        beam_size=beam, vocab_size=len(tokens),
        weights={"decoder": 1.0 - ctc_weight, "ctc": ctc_weight, "lm": 0.0, "length_bonus": 0.0},
        scorers=scorers, sos=model.odim - 1, eos=model.odim - 1, token_list=tokens,
        pre_beam_score_key="full",
    ).to(device)

    def run(video):
        x = model.frontend(video.unsqueeze(1).unsqueeze(0))  # (1, T, 1, 88, 88)
        x = model.proj_encoder(x)
        feat, _ = model.encoder(x, None)
        feat = feat.float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_enc = time.perf_counter()
        hyps = search(feat.squeeze(0))
        ids = torch.tensor(list(map(int, hyps[0].asdict()["yseq"][1:])))
        return text_transform.post_process(ids).replace("<eos>", "").strip(), t_enc

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=MODELS)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/lrs3-test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="first N clips only")
    p.add_argument("--beam", type=int, default=40)
    p.add_argument("--ctc-weight", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true", help="autocast the encoder and decoder to fp16")
    p.add_argument("--half", action="store_true", help="store the encoder weights in fp16 (decoder and CTC stay fp32)")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.float16 if args.half else torch.float32
    family, backbone, ckpt_name = MODELS[args.model]
    ckpt = CKPT / ckpt_name
    if family == "usr2":
        run = load_usr2(backbone, ckpt, args.beam, args.ctc_weight, device, dtype)
    else:
        run = load_autoavsr(ckpt, args.beam, args.ctc_weight, device, dtype)
    vram_model = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()

    clip_ids = sorted(p.stem for p in (args.dataset / "mouths").glob("*.npy"))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    # Results are appended per clip so a crash (e.g. OOM) can resume where it stopped.
    hyps_path, timing_path = args.out / "hyps.tsv", args.out / "timing.tsv"
    done = set()
    if timing_path.exists():
        done = {line.split("\t")[0] for line in timing_path.read_text().splitlines()[1:]}
    else:
        timing_path.write_text("clip_id\tvideo_s\tencode_s\tdecode_s\tstatus\n")
        hyps_path.write_text("")
    with hyps_path.open("a") as hyps_f, timing_path.open("a") as timing_f:
        for i, clip_id in enumerate(clip_ids):
            if clip_id in done:
                continue
            frames = np.load(args.dataset / "mouths" / f"{clip_id}.npy")
            video = TRANSFORM(torch.from_numpy(frames).float().div(255).to(device)).to(dtype)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16):
                    text, t_enc = run(video)
                torch.cuda.synchronize()
                status = "ok"
            except torch.OutOfMemoryError:
                # Scored as an empty Transcript; the count goes in the summary.
                del video
                torch.cuda.empty_cache()
                text, t_enc, status = "", time.perf_counter(), "oom"
            t1 = time.perf_counter()
            hyps_f.write(f"{clip_id}\t{text}\n")
            timing_f.write(f"{clip_id}\t{len(frames) / 25:.2f}\t{t_enc - t0:.4f}\t{t1 - t_enc:.4f}\t{status}\n")
            hyps_f.flush()
            timing_f.flush()
            if i % 50 == 0:
                print(f"[{i}/{len(clip_ids)}] {clip_id}: {text}", flush=True)

    all_rows = [line.split("\t") for line in timing_path.read_text().splitlines()[1:]]
    rows = [r for r in all_rows if r[4:] != ["oom"]]  # speed is measured on decoded clips only
    total_video = sum(float(r[1]) for r in rows)
    total_enc = sum(float(r[2]) for r in rows)
    total_dec = sum(float(r[3]) for r in rows)
    total_video = total_video or float("nan")  # every clip OOMed
    summary = dict(
        model=args.model, beam=args.beam, ctc_weight=args.ctc_weight, fp16=args.fp16, half=args.half,
        clips=len(all_rows), oom_clips=len(all_rows) - len(rows), video_seconds=round(total_video, 1),
        encode_rtf=round(total_enc / total_video, 4), decode_rtf=round(total_dec / total_video, 4),
        rtf=round((total_enc + total_dec) / total_video, 4),
        vram_weights_mib=round(vram_model), vram_peak_mib=round(torch.cuda.max_memory_allocated() / 2**20),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
