from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, classification_report
from rapidfuzz import process, fuzz
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import wandb

# === Match predicted concepts to GT concepts using fuzzy string similarity ===
def match_concepts(gt_dict, pred_dict):
    matched_pairs = []
    
    if not gt_dict or not pred_dict:
        return matched_pairs

    gt_concepts = list(gt_dict.keys())

    for pred_concept, pred_cat in pred_dict.items():
        if gt_concepts:
            result = process.extractOne(pred_concept, gt_concepts, scorer=fuzz.token_sort_ratio)
            if result is not None:
                best_match, score, _ = result
                matched_pairs.append((gt_dict[best_match], pred_cat, score))

    return matched_pairs

# === Evaluate per sample ===
def evaluate_sample_fuzzy(gt_concepts, pred_concepts):
    matched = match_concepts(gt_concepts, pred_concepts)

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
        "y_pred": y_pred
    }


# === Init wandb ===
wandb.init(project="qwen3-medical-extraction", name="grpo_qwen3-concept_categorization_lora_multi_adapter")


gt_samples, pred_samples = [], []
import json
with open('/media/shrish/Data/medpao_fast/Final_gt.json') as f:
    data = json.load(f)

with open('/media/shrish/Data/medpao_fast/predicted_default.json') as f:
    data1 = json.load(f)

for d in data:   
    for d1 in data1:
        if d['id'] == str(d1['id']):
            gt_samples.append(d['concept_categorization'])
            pred_samples.append(d1['predicted_categorized_concepts'])

# === Evaluate all ===
results = []
for i, (gt, pred) in enumerate(zip(gt_samples, pred_samples)):
    r = evaluate_sample_fuzzy(gt, pred)
    if r is not None:
        results.append(r)
        matches = match_concepts(gt, pred)
        print(f"Sample {i+1}: {len(matches)} matches found")
        for j, (true_cat, pred_cat, score) in enumerate(matches):
            print(f"  Match {j+1}: {true_cat} -> {pred_cat} (similarity: {score:.1f})")
    else:
        print(f"Sample {i+1}: No matches found")

# === Average metrics ===
if results:
    averaged = defaultdict(list)
    all_y_true = []
    all_y_pred = []
    
    for r in results:
        for k, v in r.items():
            if k not in ['y_true', 'y_pred']:
                averaged[k].append(v)
        all_y_true.extend(r['y_true'])
        all_y_pred.extend(r['y_pred'])

    final_scores = {k: np.mean(v) for k, v in averaged.items()}

    # === Print averaged metrics ===
    print("\n=== Fuzzy Concept Matching Evaluation (Per-Sample Averaged) ===")
    print(f"Total samples evaluated: {len(results)}/{len(gt_samples)}")
    for k in sorted(final_scores.keys()):
        print(f"{k}: {final_scores[k]:.3f}")

    # === Log all scalar scores to wandb ===
    wandb.log({
        **{k: v for k, v in final_scores.items()},
        "total_samples_evaluated": len(results),
        "total_samples": len(gt_samples),
    })

    # === Generate Overall Confusion Matrix ===
    print("\n=== Overall Confusion Matrix ===")
    labels = sorted(list(set(all_y_true + all_y_pred)))
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
    
    cm_df = pd.DataFrame(cm, index=[f"True_{label}" for label in labels], 
                        columns=[f"Pred_{label}" for label in labels])
    print(cm_df)
    
    # === Raw Confusion Matrix Heatmap ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f"Pred_{label}" for label in labels],
                yticklabels=[f"True_{label}" for label in labels],
                cbar_kws={'label': 'Count'})
    
    plt.title('Aggregated Confusion Matrix\n(Fuzzy Concept Matching)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Categories', fontsize=12, fontweight='bold')
    plt.ylabel('True Categories', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_aggregated.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix heatmap saved as 'confusion_matrix_aggregated.png'")
    plt.show()
    
    # === Normalized Confusion Matrix Heatmap ===
    fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=[f"Pred_{label}" for label in labels],
                yticklabels=[f"True_{label}" for label in labels],
                cbar_kws={'label': 'Percentage'},
                ax=ax_norm)
    
    ax_norm.set_xlabel('Predicted Categories', fontsize=12, fontweight='bold')
    ax_norm.set_ylabel('True Categories', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✅ Normalized confusion matrix heatmap saved as 'confusion_matrix_normalized.png'")

    # === Log normalized confusion matrix figure to wandb ===
    wandb.log({"confusion_matrix_normalized": wandb.Image(fig_norm, caption="Normalized Confusion Matrix (Fuzzy Concept Matching)")})
    plt.show()
    
    # === Generate Classification Report ===
    print("\n=== Classification Report ===")
    report_str = classification_report(all_y_true, all_y_pred, labels=labels, zero_division=0)
    print(report_str)

    # Log classification report as a wandb text artifact
    report_dict = classification_report(all_y_true, all_y_pred, labels=labels, zero_division=0, output_dict=True)
    # Flatten per-class metrics into wandb-friendly keys
    for class_label, class_metrics in report_dict.items():
        if isinstance(class_metrics, dict):
            for metric_name, metric_val in class_metrics.items():
                safe_label = str(class_label).replace(" ", "_")
                wandb.log({f"cls_report/{safe_label}/{metric_name}": metric_val})

    # === Per-Sample Confusion Matrices ===
    print("\n=== Per-Sample Confusion Matrices ===")
    for i, r in enumerate(results):
        if r['y_true'] and r['y_pred']:
            print(f"\nSample {i+1}:")
            sample_labels = sorted(list(set(r['y_true'] + r['y_pred'])))
            sample_cm = confusion_matrix(r['y_true'], r['y_pred'], labels=sample_labels)
            sample_cm_df = pd.DataFrame(sample_cm, 
                                      index=[f"True_{label}" for label in sample_labels],
                                      columns=[f"Pred_{label}" for label in sample_labels])
            print(sample_cm_df)

    wandb.finish()

else:
    print("\n=== No valid matches found across all samples ===")
    print("Check your input data for any issues.")
    wandb.finish()