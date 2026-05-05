import os
import time

os.environ['HF_HOME'] = '/media/shrish/Data/huggingface_models'

import json
import time
from tqdm import tqdm
from langgraph_agent import full_agent



if __name__ == "__main__":

    with open("Final_gt.json", 'r') as f:
        gt = json.load(f)

    predicted=[]
    for item in tqdm(gt):
        report = item['report']

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
            "user_query": "the task is to structure the given medical report according to ABCDEF protocol",
        })

        element = {
            "id": item['id'],
            "report": report,
            "predicted concepts": ','.join(list(out1['concepts'].keys())),
            "predicted_categorized_concepts": {**out1['existing_categorized_concepts'], **out1['new_categorized_concepts']},
            "predicted_structured_report": out1['structured_report'],
        }

        predicted.append(element)

    with open("predicted_new.json", 'w') as f:
        json.dump(predicted, f, indent=4)


