# CAD-DiffQA-datasets

Data and evaluation artifacts for a CAD edit-description task: given a pair of
CAD models (original and edited), a model must describe the geometric edit in
natural language. This repo contains the benchmark data, model predictions
under several input conditions, LLM-judge evaluation results, and
human-agreement (Cohen's kappa) analysis.

## Folder structure

### `benchmark/`
Raw CAD assets backing the benchmark.
- `cad_imgs/org/`, `cad_imgs/edit/` — per-sample subfolders with the original
  and edited CAD model files (`.step`, `.stl`, `.obj`).
- `cad_imgs/org_imgs/`, `cad_imgs/edit_imgs/` — rendered multi-view PNGs
  (`front`, `iso`, `right`, `top`) of the original and edited models.

### `dataset.json`
2,000 (original, edited) CAD sequence pairs. Each entry has the token-level
CAD sequences (`original_sequence`, `edited_sequence`), the edit `type`
(`add` / `delete` / `modify`), the source `method` (`sequence` or `visual`),
and a natural-language edit `instruction`.

### `ground_truth_500.json`
500 gold edit descriptions (`index`, `answer`) used as ground truth when
scoring model predictions.

### `evaluate_gpt.py`
Scores a model's predicted edit descriptions against ground truth using
GPT-5-mini as an LLM judge, across two criteria groups:
- **visual**: `operation_type`, `affected_feature`, `spatial_location`
- **program_diff**: `parameter_identity`, `feature_binding`

Each criterion is labeled `match` / `partial` / `conflict` / `none` and
reduced to a 0/1 score.

### `analyze_cad_images.py`
Generates a model's predicted edit descriptions for `dataset.json` pairs by
querying a vision-language model (default: Qwen2.5-VL-72B via an
OpenAI-compatible endpoint). Supports four prompt strategies via
`--prompt-mode`:
- **visual-only** — 8 multi-view images only, no CAD sequences.
- **program-only** — CAD sequences only, no images (text-only call).
- **joint** — original vs. edited images and sequences presented together.
- **text-bridge** (default) — sequence diff and image diff computed
  separately, then combined into a final answer.

Output is a JSON file of `{index, original_pic_name, edited_pic_name,
original_sequence, edited_sequence, original_images, edited_images,
question, answer}` records, matching the format consumed by
`evaluate_gpt.py`.

### `Claude/`, `Qwen/`
Per-model predictions and evaluation results, one pair of subfolders per
model:
- `test_<model>/test_<model>_{early,img,late,seq}.json` — the model's
  predicted edit description (`answer`) for each of 4 input conditions:
  `early` (early fusion of image + sequence), `img` (image-only), `late`
  (late fusion), `seq` (sequence-only).
- `evaluation/eval_results_<model>_<condition>_criteria.json` — the
  GPT-judge output (via `evaluate_gpt.py`) scoring those predictions against
  `ground_truth_500.json`. Qwen's `early`/`late` conditions have `v1`/`v2`
  rerun variants.

### `Kappa/`
Human-vs-LLM-judge agreement analysis on a 50-sample subset.
- `eval_human_binary.json` — human binary judgments of Qwen's and Gemma's
  predicted descriptions against ground truth.
- `eval_results_qwen_criteria.json`, `eval_results_gemma_criteria.json` —
  the corresponding LLM-judge scores for that same subset.
- `compute_kappa.py` — computes weighted Cohen's kappa between the human
  labels and the LLM-judge labels, per criterion, to quantify how well the
  LLM judge agrees with human raters.
