"""
Fuzzy concept-categorization evaluation script.

All paths, wandb settings, and output locations are configurable via CLI args.
Run `python evaluate_abcde.py --help` to see all options.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
    classification_report,
)
from rapidfuzz import process, fuzz


# ------------------------------------------------------------------------- #
# CLI configuration
# ------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted concept categorization against ground truth "
                     "using fuzzy string matching, with metrics logged to wandb."
    )

    # --- Input data paths ---
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="Path to ground-truth JSON file (e.g. Final_gt.json).",
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        required=True,
        help="Path to predictions JSON file (e.g. predicted_default.json).",
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        default="concept_categorization",
        help="Key in each GT record holding the concept categorization dict.",
    )
    parser.add_argument(
        "--pred-key",
        type=str,
        default="predicted_categorized_concepts",
        help="Key in each prediction record holding the predicted concepts dict.",
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="id",
        help="Key used to match GT and prediction records by ID.",
    )

    # --- Fuzzy matching ---
    parser.add_argument(
        "--fuzzy-scorer",
        type=str,
        default="token_sort_ratio",
        choices=["token_sort_ratio", "ratio", "partial_ratio", "token_set_ratio"],
        help="rapidfuzz scorer to use for concept name matching.",
    )

    # --- wandb configuration ---
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="qwen3-medical-extraction",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="grpo_qwen3-concept_categorization_lora_multi_adapter",
        help="wandb run name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity (team/user). Defaults to wandb's configured default.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging entirely (useful for local/dry runs).",
    )

    # --- Output paths ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output plots (confusion matrices, etc.).",
    )
    parser.add_argument(
        "--cm-raw-filename",
        type=str,
        default="confusion_matrix_aggregated.png",
        help="Filename for the raw (count) confusion matrix heatmap.",
    )
    parser.add_argument(
        "--cm-normalized-filename",
        type=str,
        default="confusion_matrix_normalized.png",
        help="Filename for the normalized confusion matrix heatmap.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating/saving/showing plots.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots but don't call plt.show() (useful for headless/CI runs).",
    )
    parser.add_argument(
        "--per-sample-cm",
        action="store_true",
        help="Print per-sample confusion matrices (verbose).",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------- #
# Core logic
# ------------------------------------------------------------------------- #
def match_concepts(gt_dict, pred_dict, scorer):
    matched_pairs = []

    if not gt_dict or not pred_dict:
        return matched_pairs

    gt_concepts = list(gt_dict.keys())

    for pred_concept, pred_cat in pred_dict.items():
        if gt_concepts:
            result = process.extractOne(pred_concept, gt_concepts, scorer=scorer)
            if result is not None:
                best_match, score, _ = result
                matched_pairs.append((gt_dict[best_match], pred_cat, score))

    return matched_pairs


def evaluate_sample_fuzzy(gt_concepts, pred_concepts, scorer):
    matched = match_concepts(gt_concepts, pred_concepts, scorer)

    if not matched:
        return None

    y_true, y_pred, scores = zip(*matched)
    y_true = list(y_true)
    y_pred = list(y_pred)
    scores = list(scores)

    return {
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),

        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),

        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "jaccard_weighted": jaccard_score(y_true, y_pred, average="weighted", zero_division=0),

        "avg_similarity_score": np.mean(scores),
        "min_similarity_score": np.min(scores),
        "max_similarity_score": np.max(scores),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    args = parse_args()

    # Resolve fuzzy scorer function from name
    scorer_map = {
        "token_sort_ratio": fuzz.token_sort_ratio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
    }
    scorer = scorer_map[args.fuzzy_scorer]

    os.makedirs(args.output_dir, exist_ok=True)
    cm_raw_path = os.path.join(args.output_dir, args.cm_raw_filename)
    cm_norm_path = os.path.join(args.output_dir, args.cm_normalized_filename)

    # === Init wandb (optional) ===
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
        )

    # === Load data ===
    with open(args.gt_path) as f:
        data = json.load(f)

    with open(args.pred_path) as f:
        data1 = json.load(f)

    gt_samples, pred_samples = [], []
    for d in data:
        for d1 in data1:
            if d[args.id_key] == str(d1[args.id_key]):
                gt_samples.append(d[args.gt_key])
                pred_samples.append(d1[args.pred_key])

    # === Evaluate all ===
    results = []
    for i, (gt, pred) in enumerate(zip(gt_samples, pred_samples)):
        r = evaluate_sample_fuzzy(gt, pred, scorer)
        if r is not None:
            results.append(r)
            matches = match_concepts(gt, pred, scorer)
            print(f"Sample {i + 1}: {len(matches)} matches found")
            for j, (true_cat, pred_cat, score) in enumerate(matches):
                print(f"  Match {j + 1}: {true_cat} -> {pred_cat} (similarity: {score:.1f})")
        else:
            print(f"Sample {i + 1}: No matches found")

    # === Average metrics ===
    if not results:
        print("\n=== No valid matches found across all samples ===")
        print("Check your input data for any issues.")
        if use_wandb:
            wandb.finish()
        return

    averaged = defaultdict(list)
    all_y_true = []
    all_y_pred = []

    for r in results:
        for k, v in r.items():
            if k not in ["y_true", "y_pred"]:
                averaged[k].append(v)
        all_y_true.extend(r["y_true"])
        all_y_pred.extend(r["y_pred"])

    final_scores = {k: np.mean(v) for k, v in averaged.items()}

    # === Print averaged metrics ===
    print("\n=== Fuzzy Concept Matching Evaluation (Per-Sample Averaged) ===")
    print(f"Total samples evaluated: {len(results)}/{len(gt_samples)}")
    for k in sorted(final_scores.keys()):
        print(f"{k}: {final_scores[k]:.3f}")

    # === Log all scalar scores to wandb ===
    if use_wandb:
        wandb.log({
            **final_scores,
            "total_samples_evaluated": len(results),
            "total_samples": len(gt_samples),
        })

    # === Generate Overall Confusion Matrix ===
    print("\n=== Overall Confusion Matrix ===")
    labels = sorted(list(set(all_y_true + all_y_pred)))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{label}" for label in labels],
        columns=[f"Pred_{label}" for label in labels],
    )
    print(cm_df)

    if not args.no_plots:
        # === Raw Confusion Matrix Heatmap ===
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Pred_{label}" for label in labels],
            yticklabels=[f"True_{label}" for label in labels],
            cbar_kws={"label": "Count"},
        )

        plt.title("Aggregated Confusion Matrix\n(Fuzzy Concept Matching)", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Categories", fontsize=12, fontweight="bold")
        plt.ylabel("True Categories", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(cm_raw_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Confusion matrix heatmap saved as '{cm_raw_path}'")
        if not args.no_show:
            plt.show()

        # === Normalized Confusion Matrix Heatmap ===
        fig_norm, ax_norm = plt.subplots(figsize=(10, 8))

        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=[f"Pred_{label}" for label in labels],
            yticklabels=[f"True_{label}" for label in labels],
            cbar_kws={"label": "Percentage"},
            ax=ax_norm,
        )

        ax_norm.set_xlabel("Predicted Categories", fontsize=12, fontweight="bold")
        ax_norm.set_ylabel("True Categories", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(cm_norm_path, dpi=300, bbox_inches="tight")
        print(f"✅ Normalized confusion matrix heatmap saved as '{cm_norm_path}'")

        if use_wandb:
            wandb.log({
                "confusion_matrix_normalized": wandb.Image(
                    fig_norm, caption="Normalized Confusion Matrix (Fuzzy Concept Matching)"
                )
            })
        if not args.no_show:
            plt.show()

    # === Generate Classification Report ===
    print("\n=== Classification Report ===")
    report_str = classification_report(all_y_true, all_y_pred, labels=labels, zero_division=0)
    print(report_str)

    if use_wandb:
        report_dict = classification_report(
            all_y_true, all_y_pred, labels=labels, zero_division=0, output_dict=True
        )
        for class_label, class_metrics in report_dict.items():
            if isinstance(class_metrics, dict):
                for metric_name, metric_val in class_metrics.items():
                    safe_label = str(class_label).replace(" ", "_")
                    wandb.log({f"cls_report/{safe_label}/{metric_name}": metric_val})

    # === Per-Sample Confusion Matrices (optional, verbose) ===
    if args.per_sample_cm:
        print("\n=== Per-Sample Confusion Matrices ===")
        for i, r in enumerate(results):
            if r["y_true"] and r["y_pred"]:
                print(f"\nSample {i + 1}:")
                sample_labels = sorted(list(set(r["y_true"] + r["y_pred"])))
                sample_cm = confusion_matrix(r["y_true"], r["y_pred"], labels=sample_labels)
                sample_cm_df = pd.DataFrame(
                    sample_cm,
                    index=[f"True_{label}" for label in sample_labels],
                    columns=[f"Pred_{label}" for label in sample_labels],
                )
                print(sample_cm_df)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()



'''
python evaluate_abcde.py \
  --gt-path /media/shrish/Data/medpao_fast/Final_gt.json \
  --pred-path /media/shrish/Data/medpao_fast/predicted_default.json \
  --wandb-project qwen3-medical-extraction \
  --wandb-run-name grpo_qwen3-concept_categorization_lora_multi_adapter \
'''