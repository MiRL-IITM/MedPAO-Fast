'''
agent_batched.py
'''

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the langgraph agent over a ground-truth report dataset "
                     "and save structured predictions to JSON."
    )

    parser.add_argument(
        "--gt-path",
        type=str,
        default="Final_gt.json",
        help="Path to the ground-truth JSON file containing 'id' and 'report' fields.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="predicted.json",
        help="Path to write the predictions JSON file to.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="/media/shrish/Data/huggingface_models",
        help="Directory to use for HF_HOME (HuggingFace cache/model directory). "
             "Must be set before any HuggingFace-dependent imports.",
    )
    parser.add_argument(
        "--user-query",
        type=str,
        default="the task is to structure the given medical report according to ABCDEF protocol",
        help="The user_query passed into the agent for every sample.",
    )
    parser.add_argument(
        "--report-key",
        type=str,
        default="report",
        help="Key in each GT record holding the report text.",
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="id",
        help="Key in each GT record holding the sample ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indent level for the output file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # HF_HOME must be set before importing anything HuggingFace-dependent
    # (including langgraph_agent, which likely pulls in transformers/etc.)
    os.environ["HF_HOME"] = args.hf_home

    import json
    from tqdm import tqdm
    from langgraph_agent import full_agent  #

    with open(args.gt_path, "r") as f:
        gt = json.load(f)

    if args.limit is not None:
        gt = gt[: args.limit]

    predicted = []
    for item in tqdm(gt):
        report = item[args.report_key]

        _base = {
            "input_report":                  report,
            "input_findings":                "",
            "modules_queue":                 [],
            "concepts":                      {},
            "existing_categorized_concepts": {},
            "new_categorized_concepts":      {},
            "ontology_mapping":              {},
            "structured_report":             {},
        }

        out1 = full_agent.invoke({
            **_base,
            "user_query": args.user_query,
        })

        element = {
            "id": item[args.id_key],
            "report": report,
            "predicted concepts": ",".join(list(out1["concepts"].keys())),
            "predicted_categorized_concepts": {
                **out1["existing_categorized_concepts"],
                **out1["new_categorized_concepts"],
            },
            "predicted_structured_report": out1["structured_report"],
        }

        predicted.append(element)

    with open(args.output_path, "w") as f:
        json.dump(predicted, f, indent=args.indent)


'''
python agent_batched.py \
  --gt-path Final_gt.json \
  --output-path predicted_multi_input_grpo.json \
  --hf-home /media/shrish/Data/huggingface_models
'''