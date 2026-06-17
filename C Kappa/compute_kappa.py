#!/usr/bin/env python3
"""Compute weighted Cohen's kappa between human annotations and LLM evaluations."""
import argparse
import json
from sklearn.metrics import cohen_kappa_score

CRITERIA = {
    "visual":       ["operation_type", "affected_feature", "spatial_location"],
    "program_diff": ["parameter_identity", "feature_binding"],
}

LABEL_TO_SCORE = {"match": 1, "partial": 1, "conflict": 0, "none": 0}


def get_human_score(item, model, category, criterion):
    """Extract 0/1 score from eval_human.json format."""
    label = item[f"{model}_scores"].get(criterion, "none")
    return LABEL_TO_SCORE.get(str(label).lower(), 0)


def get_llm_score(item, category, criterion):
    """Extract 0/1 score from eval_results_gemma/qwen.json format."""
    val = item.get(category, {}).get(criterion, {})
    if isinstance(val, dict):
        return val.get("score", 0)
    return LABEL_TO_SCORE.get(str(val).lower(), 0)


def compute_kappa(human_path, llm_path, model):
    with open(human_path) as f:
        human_data = json.load(f)
    human_results = human_data.get("results", human_data) if isinstance(human_data, dict) else human_data

    with open(llm_path) as f:
        llm_data = json.load(f)
    llm_results = llm_data.get("results", llm_data) if isinstance(llm_data, dict) else llm_data

    human_by_index = {item["index"]: item for item in human_results}
    llm_by_index   = {item["index"]: item for item in llm_results}
    common = sorted(set(human_by_index) & set(llm_by_index))

    print(f"Model: {model}  |  {len(common)} samples")
    print(f"Human: {human_path}")
    print(f"LLM:   {llm_path}")
    print()

    all_kappas = []
    for category, keys in CRITERIA.items():
        print(f"[{category}]")
        for criterion in keys:
            human_scores = [get_human_score(human_by_index[i], model, category, criterion) for i in common]
            llm_scores   = [get_llm_score(llm_by_index[i], category, criterion) for i in common]

            # kappa requires at least 2 distinct values in at least one rater
            if len(set(human_scores)) < 2 and len(set(llm_scores)) < 2:
                print(f"  {criterion:25s}: κ = N/A (no variance)")
                continue

            kappa = cohen_kappa_score(human_scores, llm_scores, weights="quadratic")
            agree = sum(h == l for h, l in zip(human_scores, llm_scores))
            print(f"  {criterion:25s}: κ = {kappa:+.3f}  (agreement {agree}/{len(common)} = {agree/len(common):.0%})")
            all_kappas.append(kappa)
        print()

    if all_kappas:
        print(f"Overall avg κ = {sum(all_kappas)/len(all_kappas):+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Weighted Cohen's kappa: human vs LLM evaluations")
    parser.add_argument("--human",  default="benchmark/eval_human.json")
    parser.add_argument("--llm",    required=True,
                        help="e.g. benchmark/eval_results_gemma.json or eval_results_qwen.json")
    parser.add_argument("--model",  required=True, choices=["gemma", "qwen"],
                        help="Which model's human scores to use from eval_human.json")
    args = parser.parse_args()

    compute_kappa(args.human, args.llm, args.model)


if __name__ == "__main__":
    main()
