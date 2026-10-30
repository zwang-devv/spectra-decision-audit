# Do Multimodal Models Decide Consistently on Equivalent Optical Spectra?

Task definitions, per-response results and analysis code for the paper *Do Multimodal Models Decide Consistently on
Equivalent Optical Spectra?* (Ziyuan Wang, Genliang Zhu, Xiaobo Chen, 2026). arXiv link: to be added.

The same measured spectrum is shown to decision models as four tables (nm, µm, scientific notation, eV) and two plots
(nm, eV), and partial spectra test whether a model abstains when the evidence needed for the answer is missing.

## Contents

- `data/tasks/<set>.jsonl`: the five test sets. The first line is a `_meta` record; every other line is one task with
  the spectra (wavelengths in nm and response values), options, answer and construction parameters.
  - `main_v1`, `suff_v3`: MOCLIP silicon-metasurface transmission spectra (comparison, extremum band, specification and
    sufficiency tasks).
  - `gold_v1`, `gold_v1t`, `gold_suff_v3`: gold-nanoparticle absorbance spectra (`gold_v1t` shows transmission).
- `data/responses.jsonl.gz`: 257,358 responses, one line per (set, batch, model, task, representation): the last
  successful response of each request.
- `data/training_logs.csv`: selected epoch and development accuracy of every LoRA adapter.
- `reproduce_tables.py`: regenerates all result tables (`out/tables/*.csv|tex`) and Figures 1–4 (`out/figures/*.pdf`).

Rendering and model-calling code, raw model outputs and adapter weights are not included; they are available from the
authors on request.

## Reproduce

```bash
pip install -r requirements.txt
python reproduce_tables.py      # about 30 s
```

With the versions in `requirements.txt`, every table is byte-identical to the paper's and the figures are
pixel-identical. Newer numpy/pandas/matplotlib give the same tables; figure rendering may differ slightly.

## Response fields

| field | meaning |
| --- | --- |
| `dataset` | test set (`main_v1`, `gold_v1`, `gold_v1t`, `suff_v3`, `gold_suff_v3`) |
| `batch` | `raw`: main runs; `raw_102e`: supplementary batch (eV row-order control, Qwen3.5-2B base, ftD/ftCc/ftDc arms) |
| `model` | see below |
| `task_id`, `item`, `type` | task, source instance, task type (`compare`, `extreme`, `spec`, `sufficiency`, `threshold`) |
| `rep` | `table_nm`, `table_um`, `table_sci`, `table_ev`, `plot_nm`, `plot_ev`; `table_ev_desc` is the eV table sorted by descending energy |
| `choice`, `answer`, `correct` | model choice, correct option, 1 if equal |
| `core` | variant shown in all six representations |
| `phrasing`, `order`, `direction` | question wording (wavelength or photon-energy axis), spectrum order, comparison direction |
| `cut`, `trunc_kind`, `truncated` | evidence deletion: `none`, `out` (outside the queried range) or `in` (inside it), and the deletion type |
| `undet` | 1 if the model answered "undetermined" |
| `pick_first` | compare/spec tasks: 1 if the model chose the first spectrum |
| `hcw` | 1 if the model gave at least 0.9 probability to a wrong option |
| `letter_mass` | Qwen3.5-2B base only: probability mass on the option letters |

Models: `d3-nano` (2.21B multimodal decision model, checkpoint `80820624a817`, local probability readout);
`d3-nano-pa` (average over forward and reversed option orders); `jev` (Jev 1.13, text only);
`luna-openrouter` (GPT-6 Luna decision route); `luna-chat`, `luna-chat-low` (GPT-6 Luna generative route with
reasoning effort none / low); `qwen35-2b` (Qwen3.5-2B base, option-letter readout); `oracle` (rule-based labels);
`d3-nano-<arm>[_s<seed>]` (LoRA adapters of d3-nano; arms are described in the adaptation part of the Results section of the paper).

## Sources and licenses

- MOCLIP silicon-metasurface spectra, Figshare, https://doi.org/10.6084/m9.figshare.30772553.v1, CC BY 4.0.
- Gold-nanoparticle UV-Vis spectra (Lachowski et al. 2022), Zenodo, https://doi.org/10.5281/zenodo.6588769, CC BY 4.0.

The task files contain spectra derived from these datasets and are released under CC BY 4.0; please also credit the
original datasets. The code is released under the MIT License (`LICENSE`).

## Citation

```bibtex
@misc{wang2026spectra,
  title  = {Do Multimodal Models Decide Consistently on Equivalent Optical Spectra?},
  author = {Wang, Ziyuan and Zhu, Genliang and Chen, Xiaobo},
  year   = {2026},
  note   = {arXiv preprint}
}
```
