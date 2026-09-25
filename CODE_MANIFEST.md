# Public code manifest

## Release identity

- Reference date: 2026-09-25.
- Authority: current paper Method and Algorithm 1; PDF/Method SHA256 hashes
  are recorded in `docs/PAPER_ALIGNMENT.md`.
- Kind: newly written standalone selection reference, not an experiment trainer.
- Public export: an exact, history-free allowlist, not a recursive repository copy.

## Included files

```text
.gitattributes
.gitignore
README.md
CODE_MANIFEST.md
requirements.txt
taylormix.py
examples/selector_demo.py
docs/PAPER_ALIGNMENT.md
docs/INTEGRATION.md
tests/test_paper_selector.py
tests/test_public_release.py
tools/public_release.py
```

`RELEASE_MANIFEST.json` is generated during export. It records relative member
names, file hashes, the standalone-reference release kind, and exclusion of
Git history. ZIP members use fixed timestamps and contain no archive comments,
extra fields, absolute paths, or symlinks.

## Excluded

Historical V13/V15 trainers and samplers, vendor runtime code, reward servers,
training and merge launchers, private configuration, internal reports, datasets,
models, checkpoints, serialized state, results, logs, caches, PDFs, manuscript
sources, Git metadata/history, author/editor tags, private paths, and credentials.
Existing development files remain local; exclusion is not a deletion or a
rewrite of development history.

## Verification boundary

The tests cover algebraic correspondence, numerical input validation, allocation
and pool policy, conditional sampling, checkpoint replay, and release hygiene.
They do not establish experiment replication, proxy calibration, empirical
coverage, runtime overhead, or training-code identity.

Third-party runtime source is not bundled in this minimal release. NumPy and
pytest are external dependencies with their own licensing terms. No new
project-wide license or upstream copyright ownership is asserted here.
