# Examples

The demo dataset is generated, not shipped, so the repository never carries a
recording of a person and never re-distributes any vendor sample file.

```bash
# from the repository root
python -m gazera_core demo --out examples/gazera_demo.tsv --subjects 10 --duration 10
python -m gazera_core analyze examples/gazera_demo.tsv --out examples/results --json > /dev/null
```

`gazera_demo.tsv` is a two-condition study (도시숲 urban forest vs 도심 가로
urban street), 10 participants, 10 s per condition, 60 Hz, with ground-truth
`Eye movement type` labels and about 2 % invalid samples. It is ideal for
checking an installation, for teaching, and for comparing settings of the I-VT
classifier against known fixations.

`examples/results/` then contains the multi-sheet Excel workbook, JSON report
and per-table CSV files.
