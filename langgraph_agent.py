import os
import time

os.environ['HF_HOME'] = '/media/shrish/Data/huggingface_models'             #set the huggingface cache directory to a custom path
CONC_EXTRACTOR_PATH = "qwen3_conc_ext/checkpoint-330"                       #set the path to the concept extractor LoRA adapter
CONC_CATEGORIZER_PATH = "grpo-qwen3-4b-cxr-multi-input/lora-adapter-final"  #set the path to the concept categorizer LoRA adapter

import torch
import re
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from ontoology_mapping import SnomedMatcher

DEVICE = 'cuda'

# ==========================================
# 1. LOAD BASE MODEL & LORA ADAPTERS
# ==========================================
print("Loading Base Qwen3 model in 4-bit mode...")
model_id = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=DEVICE,
)

print("Model loaded successfully")
print("USING DEVICE:", base_model.device)
TESTING_MODE = False

print("Initializing SNOMED Matcher...")
matcher = SnomedMatcher()


if TESTING_MODE:
    print("Testing Mode: No LoRAs loaded.")
    model = base_model
else:
    print("Loading LoRA Adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        CONC_EXTRACTOR_PATH,
        adapter_name="get_concept",
    )
    model.load_adapter(CONC_CATEGORIZER_PATH, adapter_name="concept_categorizer_multi")


# ==========================================
# 2. INFERENCE HELPERS
# ==========================================
def generate_with_adapter(msgs: list, adapter_name: str, max_input_length: int = 2048, max_new_tokens: int = 650) -> str:
    """Text-only inference with LoRA hot-swap."""
    prompts = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        [prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    # print("Input to model =", prompts)
    print(f"      [LLM Generation Started - Max Tokens: {max_new_tokens}]")

    _gen_start = time.time()

    if adapter_name == "default" or TESTING_MODE:
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
        with ctx:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
    else:
        model.set_adapter(adapter_name)
        print(f"      [Active LoRA swapped to: {adapter_name}]")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    prompt_len = inputs["input_ids"].shape[1]
    # FIX 1: was `outputs[j]` — `j` was never defined; correct index is 0
    generated = outputs[0][prompt_len:]
    pred_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    _gen_elapsed = time.time() - _gen_start
    print(f"      [LLM Generation Complete — {_gen_elapsed:.2f}s]")
    # print("*"*50)
    # print("Generated raw response:", pred_text)

    # print("*" * 50)
    return pred_text


# FIX 2: `parse_response` was completely empty (just `pass`).
# Implemented per-tool parsing logic below.
def parse_response(model_response: str, tool_name: str):
    """
    Parses raw LLM text into the structured type each tool expects.

    get_concept            → List[str]   (comma-separated concepts)
    categorize_concepts    → dict[str, str]
    generate_structured_report → dict[str, str]
    """
    if tool_name == "get_concept":
        # Model returns "CONCEPT1, CONCEPT2, CONCEPT3"
        # Strip any think-blocks Qwen3 might emit even with do_sample=False
        cleaned = re.sub(r"<think>.*?</think>", "", model_response, flags=re.DOTALL).strip()
        concepts = [c.strip() for c in cleaned.split(",") if c.strip()]
        return concepts

    elif tool_name in ("categorize_concepts", "generate_structured_report"):
        # Model returns a JSON object
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", model_response).replace("```", "").strip()
        # Also strip think-blocks
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        # Find the first JSON object in the output
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  [parse_response] JSON decode failed for {tool_name}, returning empty dict")
                return {}
        print(f"  [parse_response] No JSON object found for {tool_name}, returning empty dict")
        return {}

    else:
        # Fallback: return raw string
        return model_response


# ==========================================
# 3. LANGGRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_query:           str
    input_report:         str
    input_findings:       Optional[str]
    modules_queue:        List[str]
    concepts:             dict[str, str]            # concept → source sentence mapping; extracted from report by get_concept tool
    existing_categorized_concepts: dict[str, str]   # from cache; merged back with new ones before categorization
    new_categorized_concepts: dict[str, str]        # from LLM categorization step; merged back with cached ones before report generation
    ontology_mapping:     dict[str, str]            # concept → ontology info (description + ancestors)
    structured_report:    dict[str, str]


# ==========================================
# 5. PLANNER NODE
# ==========================================
MODULE_DESCRIPTIONS = {
    "get_concept":               "Get the medical concepts from the given medical report.",
    "check_cache":               "Check if extracted concepts are already in local vocabulary.",
    "ontology_mapping":          "Get SNOMEDCT/RADLEX ontology mapping for each concept.",
    "categorize_concepts":       "Categorize concepts into ABCDEF protocol buckets.",
    "generate_structured_report":"Generate the final structured report from categorized concepts.",
}


def planner_node(state: AgentState):
    print("\n--- NODE: Planner (Dynamic Module Selection) ---")

    system_prompt = (
        "You are a strict medical AI pipeline orchestrator. "
        "You must respond with ONLY a valid JSON array of tool names. "
        "NO markdown formatting, NO explanations, NO extra text. "
        "OUTPUT FORMAT EXACTLY LIKE THIS: [\"TOOL1\", \"TOOL2\"]"
    )

    prompt = f"""TOOL REGISTRY:
            - get_concept: (extracting concepts from report) 
            - check_cache: (checking if concepts are previously processed) 
            - ontology_mapping: (getting ontology information per concept) 
            - categorize_concepts: (categorize the concepts into ABCDEF categories) 
            - generate_structured_report: (generating structured report based on concepts)

            TASK: Select which tools are needed for the query below, in execution order.

            EXAMPLE QUERY: "structure this radiology report according to protocol"
            EXAMPLE OUTPUT: ["get_concept", "check_cache", "ontology_mapping", "categorize_concepts", "generate_structured_report"]

            EXAMPLE QUERY: "map these concepts to ontologies: consolidation, effusion"
            EXAMPLE OUTPUT: ["ontology_mapping"]

            NOW ANSWER:
            Report snippet: "{str(state['input_report'])}"
            Query: "{state['user_query']}"
            """

    final_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    # Pass max_new_tokens=60 to restrict the OUTPUT length, not the input!
    raw = generate_with_adapter(final_prompt, adapter_name="default", max_new_tokens=60)

    # Strip any markdown blocks the model might try to wrap the JSON in
    raw = raw.replace("```json", "").replace("```", "").strip()

    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            selected = json.loads(match.group())
            modules_queue = selected
        except json.JSONDecodeError as e:
            print(f"  [Planner] JSON Decode Error: {e} — running all modules as fallback.")
            modules_queue = ["get_concept", "check_cache", "ontology_mapping", "categorize_concepts", "generate_structured_report"]
    else:
        print("  [Planner] Could not parse module list — running all modules as fallback.")
        modules_queue = ["get_concept", "check_cache", "ontology_mapping", "categorize_concepts", "generate_structured_report"]

    print(f"  [Planner] Selected modules: {modules_queue}")
    return {"modules_queue": modules_queue}


# ==========================================
# 6. ROUTING LOGIC
# ==========================================
def route_next(state: AgentState) -> str:
    queue = state.get("modules_queue", [])
    if not queue:
        return "end"        # route to END directly; no orchestrator needed
    return queue[0]


# ==========================================
# 7. MODULE NODES
# ==========================================
def _pop_queue(state: AgentState) -> List[str]:
    return state.get("modules_queue", [])[1:]


def run_get_concept_tool(state: AgentState):
    print("\n--- TOOL: get_concept ---")

    # Example reports used to demonstrate concept extraction to the model
    rep1 = 'No focal consolidation, pleural effusion, or pneumothorax is seen. Heart and mediastinal contours are within normal limits.  There is no evidence for pulmonary edema.'
    rep2 = 'PA and lateral views of the chest provided.   Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax. The cardiomediastinal silhouette is notable for an unfolded  thoracic aorta.  Imaged osseous structures are intact.  No free air below the  right hemidiaphragm is seen.'
    rep3 = "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.  No new consolidations are identified.  There is no  pleural effusion or pneumothorax.  Cardiac and mediastinal silhouettes are  normal.  No acute fractures are identified."
    rep4 = " The lungs are clear with no evidence of consolidation, effusion, or  pneumothorax.  Lung volumes are low.  Cardiomediastinal silhouette is normal.   No acute fractures are identified.   "
    
    # Example concepts mapped to their source sentences for each report
    conc1 = {
        "no focal consolidation" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "no pleural effusion" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "no pneumothorax" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "normal heart contours" : "Heart and mediastinal contours are within normal limits.", 
        "normal mediastinal contours" : "Heart and mediastinal contours are within normal limits.", 
        "no pulmonary edema" : " There is no evidence for pulmonary edema."
    }


    conc2 = {
    "PA and lateral views": "PA and lateral views of the chest provided.",
    "low lung volumes": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no focal consolidation": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no effusion": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no pneumothorax": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "noteable cardiomediastinal silhouette": "The cardiomediastinal silhouette is notable for an unfolded  thoracic aorta. ", 
    "intact osseous structures": "Imaged osseous structures are intact.",
    "no free air below right hemidiaphragm": " No free air below the  right hemidiaphragm is seen."
    }

    conc3 = {
        "improving left lower lobe opacity":"Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "likely due to resolving pneumonia": "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "no new consolidations": "No new consolidations are identified.",
            "no pleural effusion": "There is no  pleural effusion or pneumothorax.",
            "normal cardiac silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "normal mediastinal silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "no acute fractures": "No acute fractures are identified."
    }

    conc4 = {
            "clear lungs": "The lungs are clear with no evidence of consolidation, effusion, or  pneumothorax.",
            "no consolidation": "The lungs are clear with no evidence of consolidation, effusion, or  pneumothorax.",
            "no effusion": "The lungs are clear with no evidence of consolidation, effusion, or  pneumothorax.",
            "no pneumothorax": "The lungs are clear with no evidence of consolidation, effusion, or  pneumothorax.",
            "low lung volumes": " Lung volumes are low.",
            "normal cardiomediastinal silhouette": "Cardiomediastinal silhouette is normal.",
            "no acute fractures": "No acute fractures are identified. "

    }
    prompt = [
        {
            "role": "system",
            "content": (
                "Assume you are an expert radiology practitioner. "
                "Extract the medical concepts and their source sentences from the given report.\n"
                "Respond with a json structure like: "
                "{{'CONCEPT1': 'source sentence 1', 'CONCEPT2': 'source sentence 2', ...}}. "
            ),
        },
        {
            "role": "user",
            "content": (f"Extracted concepts and their corresponding sentence from '{rep1}' are {conc1} ##\n"
                        f"Extracted concepts and their corresponding sentence from '{rep2}' are {conc2} ##\n"
                        f"Extracted concepts and their corresponding sentence from '{rep3}' are {conc3} ##\n"
                        f"Extracted concepts and their source sentences from '{state['input_report']}' are :"),
        },
    ]
    result = generate_with_adapter(prompt, adapter_name="get_concept", max_new_tokens=300)

    
    #parse the result to load a json in concepts;
    concepts = json.loads(result)

    print(f"  Extracted concepts: {concepts}")
    return {"concepts": concepts, "modules_queue": _pop_queue(state)}


def run_check_cache_tool(state: AgentState):
    """
    Checks which concepts are already in the local vocabulary cache.
    Cached ones skip ontology mapping; new ones go through it.

    FIX 3: was returning a plain string "Done" instead of a proper state dict,
    and was never popping itself from modules_queue.
    """
    print("\n--- TOOL: check_cache ---")
    _gen_start = time.time()
    cache_path = os.path.join(os.getcwd(), 'cached_vocab.json')

    if not os.path.exists(cache_path):
        print("  [Cache] No cache file found. All concepts will need mapping.")
        _gen_elapsed = time.time() - _gen_start
        print(f"      [Check cache tool Complete — {_gen_elapsed:.2f}s]")
        return {
            "existing_categorized_concepts": {},  # pass all concepts forward
            "modules_queue": _pop_queue(state),
        }
    
    with open(cache_path) as f:
        vocab = json.load(f)

    concepts = state.get("concepts", [])

    cached    = {c: vocab[c] for c in concepts if c in vocab}
    uncached  = [c for c in concepts if c not in vocab]

    print(f"  Cached: {list(cached.keys())}")
    print(f"  Needs mapping: {uncached}")

    _gen_elapsed = time.time() - _gen_start
    print(f"      [Check cache tool Complete — {_gen_elapsed:.2f}s]")
    return {
        "existing_categorized_concepts": cached,
        "modules_queue":      _pop_queue(state),
    }



def run_ontology_mapping_tool(state: AgentState):
    print("\n--- TOOL: ontology mapping ---")
    _gen_start = time.time()
    concepts_to_map = [c for c in state["concepts"].keys()
                       if c not in state.get("existing_categorized_concepts", {}).keys()]

    if len(concepts_to_map)==0:
        print("  [Ontology Mapping] No new concepts to map. Skipping.")
        _gen_elapsed = time.time() - _gen_start
        print(f"      [ontology mapping tool Complete — {_gen_elapsed:.2f}s]")
        return {
            "ontology_mapping": {},  # pass empty mapping forward
            "modules_queue": _pop_queue(state),
        }
    
    # Single batched call — encoder loads, encodes all, then unloads
    all_matches = matcher.match(concepts_to_map, top_k=1)

    concept_ontology_map = {}
    for matches in all_matches:                     # one inner list per concept
        m = matches[0]                              # top_k=1 so first result
        ancestors = m['ancestors'][:5]
        ancestors_str = "\nAncestors: " + ', '.join(f['desc'] for f in ancestors)
        concept_ontology_map[m['query']] = "Description: " + str(m['preferred_desc']) + ancestors_str

    print(f"  Ontology map: {concept_ontology_map}")
    _gen_elapsed = time.time() - _gen_start
    print(f"      [ontology mapping tool Complete — {_gen_elapsed:.2f}s]")
    return {"ontology_mapping": concept_ontology_map, "modules_queue": _pop_queue(state)}

import os
import json

def run_categorize_concepts_tool(state: AgentState):
    print("\n--- TOOL: categorize_concepts ---")

    ontology_mapping = state.get("ontology_mapping", {})
    if not ontology_mapping:
        print("  [Categorization] No new concepts to categorize. Skipping.")
        return {
            "categorized_concepts": {},
            "modules_queue": _pop_queue(state),
        }
    
    # The system message remains exactly the same, keeping your critical overrides intact.
    sys_msg = '''You are a strict radiology expert and medical classifier.
        Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:

        A: airways (intraluminal mass, narrowing, splayed carina)
        B: breathing (lungs, pulmonary vessels, pleural spaces)
        C: circulation (cardiomediastinal contour, great vessels)
        D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)
        E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues
        F: foreign material (devices, foreign bodies, gossypibomas)

        Airways
        Start at the top in the midline and review the airways.
            trace the trachea down to the carina and main bronchi
                the trachea should be midline at the sternal notch, deviates to the right around the aortic arch and divides into the right and left main bronchi with an angle  less than 105’ (mean 80’)
                any narrowing or intraluminal lesion?
            trace down both main bronchi
                is the carina wide (more than 105 degrees)?
                is there bronchial narrowing or cut-off?
                is there any inhaled foreign body?

        Breathing
        Look for lung and pleural pathology.
            both lungs should be well expanded and similar in volume
                can you count 10 posterior ribs bilaterally?
                is one lung larger than the other?
            compare the apical, upper, middle and lower zones in turn
                are they symmetrical?
                are there areas of increased density?
            trace the lung vessels
                can you see the vasculature equally throughout both lungs?
                can you see the retrocardiac and retrodiaphragmatic lung vessels?
                are there extra lines in the periphery that aren't vessels?
            trace the lateral margins of the lung to the costophrenic angles
                are the costophrenic angles crisp?
            trace the hemidiaphragms to the vertebrae
                can you see the whole of the hemidiaphragm?
            trace the cardiac borders
                can you clearly see the left and right heart borders?
                can you see the descending aorta?
            check the heart shadow for retrocardiac lung opacity
            check the diaphragm for overlying lung lesions in the posterior costophrenic recesses

        Circulation
        Look at the heart and vessels (systemic and pulmonary).
            check the cardiac position
                is 1/3 to the right and 2/3 to the left?
            assess cardiac size
                is the cardiothoracic ratio <50%?
            check the position and size of the aortic arch and pulmonary trunk
            check the width of the upper mediastinum
            look at the hilar vessels
                can you see them clearly on both sides?
                are they at a similar height?
                can you see a preserved hilar point bilaterally and a little higher on the left? 

        Diaphragm and below
            is the right hemidiaphragm the same height or up to 2 cms higher than the left hemidiaphragm?
            is there a hiatus hernia?
            can you identify the gastric bubble, splenic flexure of the colon and spleen?
            is there any free intraperitoneal gas?
            is the stomach or bowel dilated?
            are there any gallbladder or renal calculi?

        External (chest wall, shoulder girdles etc)
        Check for any bone pathology (fracture or metastasis) and soft tissue symmetry
            equal companion shadows superior to both clavicles
            symmetrical or left slightly lower breast shadows?
            soft tissue emphysema?
            trace along each posterior (horizontal) rib on one side of the chest
                is there a fracture or area of destruction?
            repeat with the other side of the chest
            now trace lateral and anterior ribs on the first side
            repeat on the other side
            now check the clavicles and shoulder girdles for bone destruction or dislocation
                can you trace around the cortex of the bones?
            check the vertebral bodies; at each level you should see two eyes (pedicles), a mouth (interlaminar space) and a nose (spinous process):
                are the bodies rectangular and of a similar height?
                can you see 2 pedicles per vertebral body?
                are there disc spaces?

        Foreign material
        Review the upper abdomen, soft tissues and chest
            are there any surgical clips?
            are there any devices?
            are lines and tubes in a satisfactory position?
            are there any unexpected foreign bodies such as retained swabs?

        ---
        CRITICAL PROTOCOL EXCEPTIONS (OVERRIDE ANATOMY):
        The radiological ABCDEF protocol takes strict precedence over anatomical hierarchies. 
        1. HILAR STRUCTURES: Any concept mentioning "hilum", "hilar", "hilar point", or "hilar vessels" MUST be categorized as C (Circulation), even if the SNOMED ontology mentions "Lung" or "Respiratory".
        2. PLEURA: Any concept mentioning "pleura", "costophrenic", "effusion", "opacity" or "pneumothorax" MUST be categorized as B (Breathing).
        3. DIAPHRAGM/ABDOMEN: Any concept mentioning "diaphragm", "hemidiaphragm", "subdiaphragmatic", "gastric", or "bowel" MUST be categorized as D (Diaphragm and below).
        4. PULMONARY VESSELS: Any concept mentioning "pulmonary vessel", "vascular", "vasculature", "pulmonary trunk" MUST be categorized as B (Breathing).
        '''

    # 1. Format all concepts and their ontology descriptions into a single block
    concepts_input = ""
    for concept, desc in ontology_mapping.items():
        concepts_input += f"- Concept: '{concept}' | Ontology: {desc}\n"

    # 2. Reframe the prompt to demonstrate batch processing
    prompt = f"""Categorize EACH of the provided concepts into ONE of the following letters: A, B, C, D, E, or F.

        EXAMPLE INPUT:
        - Concept: 'normal hilar contours' | Ontology: Structure of hilum of lung (body structure). Ancestors: Lobe of lung, Lung, Respiratory system.
        - Concept: 'blunted costophrenic angle' | Ontology: Costodiaphragmatic recess. Ancestors: Pleura, Thorax, Respiratory system.

        EXAMPLE OUTPUT:
        {{
        "normal hilar contours": "C",
        "blunted costophrenic angle": "B"
        }}

        NOW PROCESS THE FOLLOWING CONCEPTS:
        {concepts_input}
        Respond ONLY with a single JSON object containing ALL concepts as keys and their assigned letter as values."""
    
    final_prompt = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": prompt},
    ]
    
    # 3. Increase max_new_tokens to accommodate the larger JSON dictionary
    # A batch of ~20 concepts will require roughly 400-600 tokens to generate safely.
    response = generate_with_adapter(final_prompt, adapter_name="default", max_new_tokens=1024)

    # 4. Your parse_response function will now return the full dictionary in one pass
    final_response = parse_response(response, tool_name="categorize_concepts")   

    print(f"  Categorized: {final_response}")

    cache_path = os.path.join(os.getcwd(), 'cached_vocab.json')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached_vocab = json.load(f)
    else:
        cached_vocab = {}

    print("Updating cached vocab.............")
    updated_cache = {**cached_vocab, **final_response}
    with open(cache_path, 'w') as f:
        json.dump(updated_cache, f, indent=4)

    print("Cache updated successfully at : ", cache_path)
    
    return {"new_categorized_concepts": final_response, "modules_queue": _pop_queue(state)}


def run_generate_structured_report_tool(state: AgentState):
    print("\n--- TOOL: generate_structured_report ---")


    print("Combining existing categorized concepts from cache with new ones from LLM...")
    existing = state.get("existing_categorized_concepts", {})
    new = state.get("new_categorized_concepts", {})
    combined_categorized_concepts = {**existing, **new}

    sys_msg = (
        "You are a radiology expert. With the following knowledge of Chest Xrays:\n"
        '''

            Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using P, A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:

                P: projection views(PA, AP, lateral views or no views specified)
                        
                A: airways (intraluminal mass, narrowing, splayed carina)

                B: breathing (lungs, pulmonary vessels, pleural spaces)

                C: circulation (cardiomediastinal contour, great vessels)

                D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)

                E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues

                F: foreign material (devices, foreign bodies, gossypibomas)

             '''
    )

    user_prompt = f'''
                You are given with the categorized medical concepts according to the above ABCDEF clinical protocol and each concept mapped to its source sentence from the text report\n
                Your task is to generate structured medical reports according to the categories mentioned above in the protocol, given the structured concepts.\n
                Your generated report must have a json format like: {{"A": "Findings of concepts belonging to A", "B": "Findings of concepts belonging to B", "C":"Findings of concepts belonging to C", "D":"Findings of concepts belonging to D", "E":"Findings of concepts belonging to E", "F":"Findings of concepts belonging to F"}}.\n
                If from the given concepts any of the categories are missing have its report as "No findings", for example if there are no concepts from "A" then have the findings as "No Findings".\n
                You must write report in a medical radiologist style, in a descriptive way, DONT JUST AGGREGATE THE CONCEPTS!, DONT BRING IN ANY EXTRA MEDICAL FINDINGS.\n
                So when the concept-source sentence mapping is:{state.get("concepts", {})} and the concept categories are: {combined_categorized_concepts}, the generated structured report is:
            '''
    final_prompt = [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_prompt},
    ]
    response = generate_with_adapter(final_prompt, adapter_name="default", max_new_tokens=500)
    result = parse_response(response, tool_name="generate_structured_report")   # returns dict
    print(f"  Structured report: {result}")
    return {"structured_report": result, "modules_queue": _pop_queue(state)}


def orchestrator_synthesis_node(state: AgentState):
    print("\n--- NODE: Orchestrator (Final Summary) ---")
    report = state.get("structured_report", {})
    print("\n=== STRUCTURED REPORT ===")
    for section, findings in report.items():
        print(f"  {section}: {findings}")
    return {}   # no state mutation needed


# ==========================================
# 8. BUILD AND COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner",                   planner_node)
workflow.add_node("get_concept",               run_get_concept_tool)
workflow.add_node("check_cache",               run_check_cache_tool)
workflow.add_node("ontology_mapping",          run_ontology_mapping_tool)
workflow.add_node("categorize_concepts",       run_categorize_concepts_tool)
workflow.add_node("generate_structured_report",run_generate_structured_report_tool)
workflow.add_node("orchestrator",              orchestrator_synthesis_node)

workflow.set_entry_point("planner")

all_modules = [
    "get_concept", "check_cache", "ontology_mapping",
    "categorize_concepts", "generate_structured_report",
    "orchestrator", "end",
]

# route_next returns "end" when queue is empty → map to END
routing_map = {m: m for m in all_modules}
routing_map["end"] = END

workflow.add_conditional_edges("planner", route_next, routing_map)
for mod in ["get_concept", "check_cache", "ontology_mapping",
            "categorize_concepts", "generate_structured_report"]:
    workflow.add_conditional_edges(mod, route_next, routing_map)

workflow.add_edge("orchestrator", END)

full_agent = workflow.compile()


# ==========================================
# 9. RUN THE AGENT
# ==========================================
if __name__ == "__main__":
    report = (
        "PA and lateral views of the chest are submitted. Lungs appear well inflated "
        "without evidence of focal airspace consolidation, pleural effusions, pulmonary "
        "edema, or pneumothorax. Cardiac and mediastinal contours are within normal limits. "
        "No acute bony abnormality is appreciated."
    )

    report = "There is a vague opacity seen within  the right lower lobe, concerning for pneumonia.  There is no pleural effusion  or pneumothorax.  The heart size is mildly enlarged.  The hilar and  mediastinal structures are unremarkable. "
    # FIX 5: concepts / ontology_mapping / categorized_concepts were initialized
    # as "" (empty string) but their TypedDict types are List / dict.
    # Corrected to proper empty defaults.
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

    print("\n=== FINAL STATE ===")
    print(out1)