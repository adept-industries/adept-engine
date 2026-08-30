# Seven-feature JIT-Fine-derived training pipeline

This offline-only pipeline trains `jitfine-expert-pr-risk-mvp-v1` from the prepared JIT-Fine
JITLine splits. It does not run in the API or worker request path.

The fixed input order is `ns`, `nd`, `nf`, `entropy`, `la`, `ld`, `fix`. Labels and feature
rows are joined one-to-one by commit hash, all source checksums are verified, and train,
validation, and test hashes must be disjoint. Thresholds come only from validation data; the
test split is evaluated once after the model and thresholds are frozen.

The prepared `fix` values are strings. The upstream helper calls `bool(x)`, which makes both
`"False"` and `"True"` true. Adept intentionally parses their meanings exactly. This is a
documented correction, not a claim that the new model reproduces the original baseline.

Entropy follows the prepared data's unnormalized base-2 Shannon contract:

```text
p(file) = (file additions + file deletions) / total changed lines
entropy = -sum(p(file) * log2(p(file)))
```

It is zero when at most one file has changed lines and cannot exceed `log2(nf)`. Runtime PR
extraction will implement the same formula in the later inference PR.

The formula was replayed exactly against three public `apache/ant-ivy` commits. The prepared
data's included file-churn vectors `[10, 30]`, `[33, 22, 2, 4]`, and
`[20, 9, 35, 5, 73]` reproduce the recorded entropy values `0.8112781244591328`,
`1.4295487875817467`, and `1.8120032760083444`. That comparison also showed that the prepared
metrics omit some non-source files present in public commit metadata. This pipeline records the
scope mismatch rather than claiming that the later all-file PR extractor is already equivalent.

Example isolated invocation from the repository root:

```bash
docker build -t adept-engine:jitfine-training .
docker run --rm --network none --read-only --tmpfs /tmp --user adept \
  --volume /absolute/path/to/adept-engine:/research-src:ro \
  --volume /absolute/path/to/data/jitline:/research-data:ro \
  --volume /absolute/path/to/artifact-output:/artifact-output \
  --workdir /research-src --env PYTHONPATH=/research-src \
  --entrypoint python adept-engine:jitfine-training \
  -m ml_training.src.train_jitfine_expert_mvp \
  --data-dir /research-data \
  --output-dir /artifact-output \
  --allow-unsafe-pickle
```

The normal production Dockerfile does not copy `ml_training/`; for a reproducibility run,
mount the repository read-only at `/research-src` and set `PYTHONPATH=/research-src`, or use a
temporary research-only Dockerfile. Generated approved files belong under
`app/risk/artifacts/jitfine-expert-pr-risk-mvp-v1/`.

Limitations are preserved in the metadata and report: training examples are commits while
runtime examples are aggregated pull requests; projects and languages can differ; and the
probability is review prioritization support, not proof of a defect.
