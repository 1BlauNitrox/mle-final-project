# Experiments

Store reproducible plans and compact result summaries here. Use one directory
per experiment:

```text
experiments/YYYY-MM-DD-short-name/
|-- README.md
|-- config.yaml
|-- summary.csv
`-- figures/
```

Do not commit raw logs, replay collections, temporary checkpoints, or large
training outputs. Record external artifact locations and checksums when those
artifacts are needed to reproduce a result.

See [`docs/0004-experimentation-protocol.md`](../docs/0004-experimentation-protocol.md).
