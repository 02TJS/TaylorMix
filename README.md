# TaylorMix

**Paper-aligned selector reference, not the original experiment trainer.**

This release implements the rollout-pre selection interface in Section 4 and
Algorithm 1 of **When Greedy Turns Risky, Can Curvature Coordinate Growth
Across Capabilities?** (ICLR 2027 review copy, checked 2026-09-25).
It uses CPU NumPy, synthetic examples, and independently testable numerical
contracts. It does not contain or claim to reproduce the paper's training
pipeline, checkpoints, datasets, benchmark results, or runtime measurements.

中文说明：这是依据当前论文 Method 与 Algorithm 1 整理的独立选择器参考实现，
不是原始实验训练器。公开包不包含训练路径、个人账号、集群启动脚本或实验产物。

## Quick start

```bash
python -m pip install -r requirements.txt
python -B examples/selector_demo.py
python -B -m pytest -p no:cacheprovider tests
```

Python 3.10+ is required. No GPU, model weights, Ray, VERL, account, or external
service is needed for the reference example and tests.

## Method

The domain layer computes normalized alignment/feedback costs and allocates
shares using `rho = softmax(log(q) - cost / allocation_temperature)`.
Largest-remainder rounding and supply-aware redistribution produce quotas.
The candidate layer forms domain-residual sketches and a shadow-anchor Gram
EMA, then freezes window-normalized relevance, learnability, and risk scores.

Within each quota:

```text
S_i = lambda_rel * normalized_relevance_i
    + lambda_learn * normalized_learnability_i
    - lambda_curv * normalized_risk_i
Gamma_i = S_i - k/(k+1) * z_i^T G mean_z - (z_i^T G z_i)/(2*(k+1))
```

Sampling uses a softmax over `Gamma`, without replacement. It retains the
paper's first-draw diagonal correction, persistent-pool removal/reset policy,
anchor exclusion, and final output shuffle. Continuous full support does not
guarantee nonzero integer quotas.

See [the equation-to-code map](docs/PAPER_ALIGNMENT.md) for the exact reviewed
PDF fingerprint, numerical conventions, and limits of the reference.
See [the integration contract](docs/INTEGRATION.md) for caller-owned feedback,
cache, model, and GRPO responsibilities.

## Public contents

- `taylormix.py`: allocation, scoring, sketches, Gram EMA, and persistent sampler.
- `examples/selector_demo.py`: a deterministic synthetic CPU example.
- `tests/`: equation, pool-policy, checkpoint, and release-hygiene regressions.
- `docs/`: paper correspondence and the external trainer boundary.
- `tools/public_release.py`: exact-allowlist export and directory/ZIP inspection.

Historical trainers, vendor runtime snapshots, internal design notes, and
cluster launchers in a development checkout are intentionally not exported.
See [CODE_MANIFEST.md](CODE_MANIFEST.md) for the distribution boundary.

## History-free release

```bash
python -B tools/public_release.py export --output ../TaylorMix-public --archive ../TaylorMix-public.zip
python -B tools/public_release.py scan ../TaylorMix-public
python -B tools/public_release.py scan ../TaylorMix-public.zip
```

Use fresh output paths. Export reads only an exact source-file allowlist,
rejects links and private content, and creates a SHA256 manifest and a ZIP
without Git history, local timestamps, owner metadata, or absolute paths.
Optional repeated `--deny-token` arguments add locally known identifiers;
neither their values nor matched content are written to reports or manifests.
Directory/ZIP scans reject unexpected members, generated files, and metadata.

Never publish a development checkout or its old Git history. Initialize any
public repository from the verified export only. Automated checks cannot
guarantee discovery of every possible identifier or secret; review the small
allowlisted file set before publication. This task does not upload anything
or choose a project license on behalf of the authors.
