#!/usr/bin/env python3
"""Evaluate CAD vision model outputs against ground truth using GPT-5-mini as judge.

Same prompts/criteria as evaluate.py, minus the "binary" dimension:
  1. visual        — operation_type, affected_feature, spatial_location
  2. program_diff  — parameter_identity, feature_binding
"""
import argparse
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompts (same wording as evaluate.py)
# ---------------------------------------------------------------------------

LABEL_RULES = """
Label definitions (apply to each criterion independently):
- match: prediction clearly and correctly satisfies the criterion
- partial: prediction is directionally correct but incomplete, vague, or uses slightly different terminology for the same thing
- conflict: prediction addresses the same criterion but states something incompatible or contradictory
- none: prediction provides no meaningful information for this criterion

Leniency rules (be generous — this is a free-form natural-language description, not a technical CAD report):
- Synonyms and paraphrases count as match (e.g. "cut" = "remove" = "subtract"; "boss" = "protrusion" = "extrusion")
- A more detailed prediction that includes the correct answer plus extra detail is still match, not partial
- Do not require exact numbers, units, or CAD-token terminology — a qualitative description consistent with the sequence diff is enough for match or partial
- Use partial when the prediction is in the right direction but only partially correct, imprecise, or described in different terms
- Use conflict only when the prediction clearly and directly contradicts the criterion (e.g. says "added" when gold says "removed")
- Use none only when the prediction says nothing that could plausibly relate to this criterion
- When unsure between two adjacent labels, pick the more generous (higher-scoring) one

Scoring: match and partial → 1; conflict and none → 0

Return ONLY valid JSON with a label per criterion."""

VISUAL_SYSTEM = """You are evaluating a CAD edit description against a ground truth and the CAD sequence diff.
The original_sequence and edited_sequence are token-level CAD programs; use them as authoritative ground truth to resolve any ambiguity in the text descriptions.

Evaluate three criteria:
1. operation_type: Does the prediction correctly identify the type of change? (e.g., add/remove/cut/fill — the extrude op in the sequence tells you the true type)
2. affected_feature: Does the prediction correctly identify which feature or region was changed? (e.g., hole, cylinder, prism, face — the sketch shape in the sequence tells you the true feature)
3. spatial_location: Does the prediction correctly describe the approximate location? (e.g., top, bottom, center, side — the extrude center coordinates in the sequence give the true location)
""" + LABEL_RULES + """

Return ONLY valid JSON:
{"operation_type": "<match|partial|conflict|none>", "affected_feature": "<match|partial|conflict|none>", "spatial_location": "<match|partial|conflict|none>"}"""

VISUAL_USER = """Ground truth: "{gold}"
Prediction: "{prediction}"
Original sequence: {original_seq}
Edited sequence: {edited_seq}"""

PROGRAM_DIFF_SYSTEM = """You are evaluating a CAD edit description against the ground truth AND the raw CAD sequence diff.
The original_sequence and edited_sequence are token-level CAD programs; the difference between them encodes the actual edit and is authoritative for all criteria.

Evaluate two criteria:
1. parameter_identity: Does the prediction correctly identify which parameter changed (e.g., radius, depth, height, width, position)? Read the sequence diff tokens to determine the true parameter, but natural-language terms that refer to the same underlying value (e.g. "size"/"diameter" for radius, "depth"/"height"/"length" for extrude distance) are acceptable — do not require the literal token name.
2. feature_binding: Is the change correctly attributed to the right feature/face/region? (e.g., correct face normal vector, correct sketch shape from the sequence)
""" + LABEL_RULES + """

Return ONLY valid JSON:
{"parameter_identity": "<match|partial|conflict|none>", "feature_binding": "<match|partial|conflict|none>"}"""

PROGRAM_DIFF_USER = """Ground truth: "{gold}"
Prediction: "{prediction}"
Original sequence: {original_seq}
Edited sequence: {edited_seq}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(client, model, system, user, max_tokens=512, retries=3):
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=120,
    )
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0.0

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            # Some models (e.g. gpt-5 family) reject these params; drop and retry.
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            if "response_format" in msg and "response_format" in kwargs:
                kwargs.pop("response_format")
                continue
            if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                continue
            attempt += 1
            if attempt >= retries:
                raise
            print(f"  [retry {attempt}/{retries}] {e}")
    return ""


def _parse_json(text):
    start = text.find('{')
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return {}
    return {}


VALID_LABELS = {"match", "partial", "conflict", "none"}


def _label_to_score(label):
    return 1 if str(label).lower() in ("match", "partial") else 0


def _invalid_entries(parsed, keys):
    """Return [(key, bad_label), ...] for any key whose label isn't one of VALID_LABELS."""
    bad = []
    for k in keys:
        val = parsed.get(k, "none")
        label = (val if isinstance(val, str) else val.get("label", "none")).lower()
        if label not in VALID_LABELS:
            bad.append((k, label))
    return bad


def _parse_criteria(parsed, keys):
    result = {}
    for k in keys:
        val = parsed.get(k, "none")
        label = (val if isinstance(val, str) else val.get("label", "none")).lower()
        if label not in VALID_LABELS:
            label = "none"
        result[k] = {"label": label, "score": _label_to_score(label)}
    # Overall score is 1 only if every criterion is match/partial; any conflict/none -> 0.
    result["score"] = 1.0 if all(result[k]["score"] == 1 for k in keys) else 0.0
    return result


def _zero_criteria(keys):
    result = {k: {"label": "none", "score": 0} for k in keys}
    result["score"] = 0.0
    return result


def _call_with_label_retry(client, model, system, user, keys, max_retries=2):
    """Call the judge and ensure every key in `keys` gets a label in VALID_LABELS.

    The judge model occasionally answers the object-level question (e.g. returns
    "remove" or "top face" for operation_type/spatial_location) instead of a
    match/partial/conflict/none judgment. If that happens, re-prompt with an
    explicit correction up to `max_retries` times before giving up (any
    remaining invalid labels are clamped to "none" by _parse_criteria).
    """
    raw = _call(client, model, system, user)
    parsed = _parse_json(raw)
    bad = _invalid_entries(parsed, keys)
    attempt = 0
    while bad and attempt < max_retries:
        attempt += 1
        note = (
            "\n\nYour previous answer used invalid value(s) for: "
            + ", ".join(f'{k}="{v}"' for k, v in bad)
            + ". Each field must be EXACTLY one of: match, partial, conflict, none "
              "-- a judgment of the prediction, NOT a description of the edit itself. "
              "Return ONLY the corrected JSON with one of those four labels per field."
        )
        raw = _call(client, model, system, user + note)
        parsed = _parse_json(raw)
        bad = _invalid_entries(parsed, keys)
    return parsed


def score_visual(client, model, gold, prediction, original_seq, edited_seq):
    keys = ["operation_type", "affected_feature", "spatial_location"]
    user = VISUAL_USER.format(gold=gold, prediction=prediction,
                               original_seq=original_seq, edited_seq=edited_seq)
    parsed = _call_with_label_retry(client, model, VISUAL_SYSTEM, user, keys)
    return _parse_criteria(parsed, keys)


def score_program_diff(client, model, gold, prediction, original_seq, edited_seq):
    keys = ["parameter_identity", "feature_binding"]
    user = PROGRAM_DIFF_USER.format(gold=gold, prediction=prediction,
                                     original_seq=original_seq, edited_seq=edited_seq)
    parsed = _call_with_label_retry(client, model, PROGRAM_DIFF_SYSTEM, user, keys)
    return _parse_criteria(parsed, keys)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def default_paths(predictions_path, eval_dir="evaluation"):
    """Derive ground-truth and output paths for a predictions file that lives in
    <model_root>/test_<model>/test_*.json, assuming the sibling layout:
        <model_root>/ground_truth/ground_truth_500.json
        <model_root>/<eval_dir>/eval_results_<suffix>_criteria.json
    e.g. LLM_Results/Claude/test_claude/test_claude_seq.json
      -> LLM_Results/Claude/ground_truth/ground_truth_500.json
      -> LLM_Results/Claude/evaluation/eval_results_claude_seq_criteria.json
    """
    pred_path = os.path.abspath(predictions_path)
    model_root = os.path.dirname(os.path.dirname(pred_path))

    ground_truth = os.path.join(model_root, "ground_truth", "ground_truth_500.json")

    stem = os.path.splitext(os.path.basename(pred_path))[0]
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    output = os.path.join(model_root, eval_dir, f"eval_results_{stem}_criteria.json")

    return ground_truth, output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    ground_truth_path,
    predictions_path,
    output_path,
    model,
    base_url,
    api_key,
    num_items=None,
    criteria=None,
):
    load_dotenv()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("No API key found. Set OPENAI_API_KEY (env or .env), or pass --api-key.")

    client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)

    with open(ground_truth_path) as f:
        gt_list = json.load(f)
    with open(predictions_path) as f:
        pred_list = json.load(f)

    gt_by_index   = {item["index"]: item for item in gt_list}
    pred_by_index = {item["index"]: item for item in pred_list}

    common_indices = sorted(set(gt_by_index) & set(pred_by_index))
    if num_items:
        common_indices = common_indices[:num_items]

    run = set(criteria or ["visual", "program_diff"])

    print("=" * 80)
    print(f"Evaluating {len(common_indices)} samples")
    print(f"  Ground truth: {ground_truth_path}")
    print(f"  Predictions:  {predictions_path}")
    print(f"  Model:        {model}")
    print(f"  Criteria:     {', '.join(sorted(run))}")
    print("=" * 80)
    print()

    VIS_KEYS  = ["operation_type", "affected_feature", "spatial_location"]
    PROG_KEYS = ["parameter_identity", "feature_binding"]

    results = []
    totals = {
        "visual":       {k: [] for k in VIS_KEYS + ["score"]},
        "program_diff": {k: [] for k in PROG_KEYS + ["score"]},
    }

    for n, idx in enumerate(common_indices, 1):
        gt_item   = gt_by_index[idx]
        pred_item = pred_by_index[idx]

        gold         = gt_item.get("answer", "")
        prediction   = pred_item.get("answer", "")
        original_seq = pred_item.get("original_sequence", "")
        edited_seq   = pred_item.get("edited_sequence", "")

        print(f"[{n}/{len(common_indices)}] index={idx}")
        print(f"  GT:   {gold[:120]}")
        print(f"  Pred: {prediction[:120] if prediction else '(blank)'}")

        try:
            if not prediction or not prediction.strip():
                vis  = _zero_criteria(VIS_KEYS)
                prog = _zero_criteria(PROG_KEYS)
            else:
                vis  = score_visual(client, model, gold, prediction, original_seq, edited_seq)       if "visual"       in run else _zero_criteria(VIS_KEYS)
                prog = score_program_diff(client, model, gold, prediction, original_seq, edited_seq) if "program_diff" in run else _zero_criteria(PROG_KEYS)
        except Exception as e:
            print(f"  ERROR: {e}")
            vis  = _zero_criteria(VIS_KEYS)
            prog = _zero_criteria(PROG_KEYS)

        def fmt(d): return f"{d['label']}({d['score']})"
        if "visual"       in run: print(f"  visual       : op={fmt(vis['operation_type'])} feat={fmt(vis['affected_feature'])} loc={fmt(vis['spatial_location'])} all={vis['score']:.0f}")
        if "program_diff" in run: print(f"  program_diff : param={fmt(prog['parameter_identity'])} bind={fmt(prog['feature_binding'])} all={prog['score']:.0f}")
        print()

        result = {"index": idx, "ground_truth": gold, "prediction": prediction}
        if "visual"       in run: result["visual"]       = vis
        if "program_diff" in run: result["program_diff"] = prog
        results.append(result)

        if "visual"       in run:
            for k in VIS_KEYS + ["score"]:
                totals["visual"][k].append(vis[k]["score"] if isinstance(vis[k], dict) else vis[k])
        if "program_diff" in run:
            for k in PROG_KEYS + ["score"]:
                totals["program_diff"][k].append(prog[k]["score"] if isinstance(prog[k], dict) else prog[k])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = {"n": len(results)}
    if "visual"       in run: summary["visual"]       = {k: avg(v) for k, v in totals["visual"].items()}
    if "program_diff" in run: summary["program_diff"] = {k: avg(v) for k, v in totals["program_diff"].items()}

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Samples evaluated : {summary['n']}")
    if "visual"       in run: print(f"  visual avg        : {summary['visual']['score']:.2%}  "
                                    f"(op={summary['visual']['operation_type']:.2f} "
                                    f"feat={summary['visual']['affected_feature']:.2f} "
                                    f"loc={summary['visual']['spatial_location']:.2f})")
    if "program_diff" in run: print(f"  program_diff avg  : {summary['program_diff']['score']:.2%}  "
                                    f"(param={summary['program_diff']['parameter_identity']:.2f} "
                                    f"bind={summary['program_diff']['feature_binding']:.2f})")
    print()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"summary": summary, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD vision model outputs with GPT-5.5-mini as judge")
    parser.add_argument("--predictions", required=True,
                        help="Path to test_*.json, e.g. LLM_Results/Claude/test_claude/test_claude_seq.json")
    parser.add_argument("--ground-truth", default=None,
                        help="Defaults to <model_root>/ground_truth/ground_truth_500.json")
    parser.add_argument("--output", default=None,
                        help="Defaults to <model_root>/<eval-dir>/eval_results_<suffix>_criteria.json")
    parser.add_argument("--eval-dir", default="evaluation",
                        help="Name of the output subfolder under <model_root> (default: evaluation)")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--num-items", type=int, default=None)
    parser.add_argument("--criteria", nargs="+", choices=["visual", "program_diff"],
                        default=None, help="Which criteria to run (default: both)")
    args = parser.parse_args()

    default_gt, default_out = default_paths(args.predictions, eval_dir=args.eval_dir)

    evaluate(
        ground_truth_path=args.ground_truth or default_gt,
        predictions_path=args.predictions,
        output_path=args.output or default_out,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        num_items=args.num_items,
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
