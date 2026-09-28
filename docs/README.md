# BTP documentation: HBM + High-Bandwidth Flash for LLM inference

This folder holds the written side of the project; the simulator code lives in the rest of the repository.

| Folder | Contents |
|---|---|
| [`report/`](report/) | `BTP_Report.tex`: the BTP report. Reproduces H3, compares four HBM/HBF arrangements, sizes the flash tier, and studies write endurance. |
| [`paper/`](paper/) | `HBF_Reliability_Paper.tex` (+ compiled PDF): research paper draft, *Read-Only Is Not Wear-Free: The Reliability Wall of High-Bandwidth Flash for LLM Inference*. Red boxes mark open research tasks. |
| [`literature/`](literature/) | Index of ~70 related papers with links, plus detailed reading notes. PDFs are not committed (copyright). |
| [`analysis/`](analysis/) | Python scripts that derive the analytical numbers used in the paper (HBF power break-even, minimum GPU count, read-disturb reclaim lifetime, Arrhenius retention, replica rotation). Standard library only. |

## Building the documents

Both `.tex` files are self-contained (embedded bibliography, TikZ/pgfplots figures):

```bash
tectonic docs/paper/HBF_Reliability_Paper.tex    # or: pdflatex, run twice
tectonic docs/report/BTP_Report.tex
```

## Reproducing the analytical numbers

```bash
python3 docs/analysis/provisioning_and_readdisturb.py
python3 docs/analysis/retention_and_replicas.py
```

Inputs are the simulator results reported in `BTP_Report.tex` (Llama-3.1-405B, 10M-token context, 32 GPUs).
