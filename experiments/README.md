# Experiments

One folder per experiment: `experiments/<name>/`, named after what it tests (e.g. `public-benchmark`, `llm-correction`). Each one has:

- `README.md`: the question (link the wayfinder ticket), how to run it, and the **results** with WER/CER from `lips-wer`. This is the record the ticket's resolution points to.
- Its own code. If it needs heavy or conflicting dependencies (a model's repo, a pinned PyTorch), give it its own `pyproject.toml` and run it with `uv run --project experiments/<name>`. Use the root project otherwise.

Inputs and outputs that are large, licensed, or contain faces go in the git-ignored `data/`, never in the experiment folder:

```
data/
  datasets/<dataset>/        # downloaded test sets (e.g. lrs3-test, wildvsr)
  recordings/                # the user's own clips (never pushed)
  runs/<experiment>/<run>/   # model outputs: hyps.tsv, report.json, logs
```

## Scoring

Every dataset gets a `refs.tsv`, and every model run writes a `hyps.tsv`. Both have one `clip_id<TAB>text` line per clip.

```sh
uv run lips-wer data/datasets/lrs3-test/refs.tsv data/runs/<experiment>/<run>/hyps.tsv \
  --json data/runs/<experiment>/<run>/report.json
```

Text is lowercased, punctuation except apostrophes is stripped, and clips missing from `hyps.tsv` count as empty Transcripts.

## Environment gotcha

This machine's shell exports ROS Humble's Python 3.10 `PYTHONPATH`, which leaks into every venv. If imports behave strangely (wrong numpy, yaml or torch), prefix the command with `PYTHONPATH=`.
