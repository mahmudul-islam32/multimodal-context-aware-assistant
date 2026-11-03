# Agent rules

Rules for any AI coding agent (or automation) working in this repository.

## Git: commits

- **No AI attribution of any kind.** Do not add `Co-Authored-By: Claude ...`,
  `Generated with Claude Code`, `Claude-Session: ...`, or any other trailer, footer,
  link or emoji that credits an AI tool or agent. This applies to commit messages,
  pull request titles and bodies, PR comments, and code comments.
- Commit as the configured repository user only; do not change `user.name` or
  `user.email`.
- Commit messages: a short imperative subject line (max ~72 characters), a blank
  line, then a body that explains *what* changed and *why*. No trailers.
- One logical change per commit. Do not mix unrelated refactors with fixes.
- Never commit secrets, `.env` files, participant/study data, raw datasets
  (`data/raw/`), runtime data (`data/runtime/`), databases or model weights.

## Git: branches and pushing

- Work on a feature branch; never push directly to `main`.
- Never force-push, rewrite or rebase history that has already been pushed
  unless the repository owner explicitly asks for it.
- Push with `git push -u origin <branch>`.

## Pull requests

- Only open a pull request when the repository owner asks for one.
- PR description: summary of the change, how it was tested, and known
  limitations. No AI attribution (see above).
- Keep PRs focused; a PR that fixes CI should only contain the fix.

## Before every commit / push

Run the same checks as CI, in a clean environment when dependencies changed:

```bash
pip install -e ".[dev]"
ruff check src tests eval scripts
pytest -q
```

- Every runtime import must be declared in `pyproject.toml` dependencies; do not
  rely on packages that happen to be installed locally.
- Never skip, disable or delete a test to make CI green; fix the cause.

## Honesty in docs and results

- Do not present synthetic or simulated data as real results; label it.
- Keep README numbers in sync with `results/metrics.json` (`mcaa-eval`).
