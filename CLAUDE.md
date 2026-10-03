# Lips

Experiments toward a tool that lip-reads silent video (live webcam first) into English text. Vocabulary lives in `CONTEXT.md`. Planning lives on the wayfinder map, GitHub issue #1.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on HyperToken9/lips (via `gh`; this `gh` is old, so use `gh api` where subcommands are missing). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Working here

- Python 3.11 via `uv`; shared code in `src/lips/`, tests via `uv run pytest`.
- Experiments live in `experiments/<name>/`, data in git-ignored `data/`. Conventions and the `lips-wer` scorer: see `experiments/README.md`.
- The shell's ROS Humble `PYTHONPATH` leaks into venvs; prefix commands with `PYTHONPATH=` if imports misbehave.
