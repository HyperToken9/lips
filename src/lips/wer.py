"""Score Transcripts against ground truth with word and character error rates.

Both files are TSVs of `clip_id<TAB>text`, one clip per line. Clips missing
from the hypothesis file are scored as empty Transcripts, so a model that
silently drops clips is penalised rather than flattered.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer


def normalize(text: str) -> str:
    """Lowercase, keep letters, digits and apostrophes, collapse whitespace."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def read_tsv(path: Path) -> dict[str, str]:
    clips: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        clip_id, sep, text = line.partition("\t")
        if not sep:
            raise ValueError(f"{path}:{line_no}: expected 'clip_id<TAB>text'")
        if clip_id in clips:
            raise ValueError(f"{path}:{line_no}: duplicate clip id {clip_id!r}")
        clips[clip_id] = text
    return clips


@dataclass
class ClipScore:
    clip_id: str
    reference: str
    hypothesis: str
    errors: int
    words: int


@dataclass
class Report:
    wer: float
    cer: float
    clips: int
    missing: int
    extra: int
    worst: list[ClipScore]


def score(refs: dict[str, str], hyps: dict[str, str], worst: int = 10) -> Report:
    ids = sorted(refs)
    ref_texts = [normalize(refs[i]) for i in ids]
    hyp_texts = [normalize(hyps.get(i, "")) for i in ids]

    # jiwer rejects empty references; such clips have no words to get wrong.
    pairs = [(i, r, h) for i, r, h in zip(ids, ref_texts, hyp_texts) if r]
    if not pairs:
        raise ValueError("no non-empty references to score")
    ids, ref_texts, hyp_texts = map(list, zip(*pairs))

    words = jiwer.process_words(ref_texts, hyp_texts)
    chars = jiwer.process_characters(ref_texts, hyp_texts)

    per_clip = []
    for clip_id, ref, hyp, alignment in zip(ids, ref_texts, hyp_texts, words.alignments):
        errors = sum(
            max(chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx)
            for chunk in alignment
            if chunk.type != "equal"
        )
        per_clip.append(ClipScore(clip_id, ref, hyp, errors, len(ref.split())))
    per_clip.sort(key=lambda c: c.errors / c.words, reverse=True)

    return Report(
        wer=words.wer,
        cer=chars.cer,
        clips=len(ids),
        missing=sum(1 for i in refs if i not in hyps),
        extra=sum(1 for i in hyps if i not in refs),
        worst=per_clip[:worst],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("refs", type=Path, help="ground-truth TSV (clip_id<TAB>text)")
    parser.add_argument("hyps", type=Path, help="model output TSV (clip_id<TAB>text)")
    parser.add_argument("--worst", type=int, default=10, help="worst clips to list")
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)

    report = score(read_tsv(args.refs), read_tsv(args.hyps), args.worst)
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2) + "\n")

    print(f"WER {report.wer:.2%}  CER {report.cer:.2%}  over {report.clips} clips")
    if report.missing or report.extra:
        print(f"missing hypotheses: {report.missing}  extra hypotheses: {report.extra}", file=sys.stderr)
    for clip in report.worst:
        print(f"\n{clip.clip_id}  ({clip.errors}/{clip.words} words wrong)")
        print(f"  ref: {clip.reference}")
        print(f"  hyp: {clip.hypothesis}")


if __name__ == "__main__":
    main()
