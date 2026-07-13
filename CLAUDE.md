# CLAUDE.md - objscale

## Agent Skill

The agent skill file lives at `objscale/SKILL.md` in this repo (inside the package, so it ships in the wheel). This is the source of truth. Never edit the installed copy in `~/.claude/skills/` directly.

**When publishing a new version**, always update the skill file to reflect any API changes (including the version number at the top), then install it locally:

```bash
cp objscale/SKILL.md ~/.claude/skills/objscale/
```

## Tests

Run all test suites from the repo root:

```bash
python tests/automated/test_wrapping.py
python tests/automated/test_size_distributions.py
python tests/automated/test_size_distributions_boundaries.py
python tests/automated/test_fractal_dimensions.py
python tests/automated/test_label_size.py
```

## Publishing

```bash
python -m build && twine upload dist/*
```
