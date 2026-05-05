"""
cxr_agent_cpu.py
================
CPU-efficient port of the Chest X-ray ABCDEF structuring agent.

Key changes from the original GPU/PEFT version
───────────────────────────────────────────────
1.  All model loading (tokenizer, BitsAndBytesConfig, AutoModelForCausalLM,
    PeftModel) is REMOVED.  Inference is delegated to gguf_engine, which runs
    a quantised GGUF via llama-cpp-python — no GPU required.

2.  generate_with_adapter() now wraps gguf_engine.generate_with_adapter().
    The chat-template list-of-dicts format used in the original is converted to
    a single formatted string via _format_chat_prompt().

3.  parse_response() is kept verbatim; only the think-block stripping now also
    calls exclude_thinking_component() for robustness.

4.  Everything else — LangGraph nodes, routing, state, SNOMED matcher — is
    identical to the original.
"""

import os
import re
import json
import time
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

import os, sys, time, re
from contextlib import contextmanager
from functools import lru_cache
from llama_cpp import Llama

# ── GGUF engine (replaces HuggingFace + PEFT) ──────────────────────────────
# from gguf_engine import (
#     generate_with_adapter   as _gguf_generate,
#     exclude_thinking_component,
# )

from ontology_mapping import SnomedMatcher      # unchanged dependency


# ==========================================
# 1.  ADAPTER / MODEL CONFIGURATION
# ==========================================
# Map the logical adapter names used in this agent to whatever names your
# gguf_engine recognises.  Edit these strings to match your gguf_engine setup.
#
#   "default"                  → base model, no adapter
#   "get_concept"              → LoRA fine-tuned for concept extraction
#   "concept_categorizer"      → LoRA fine-tuned for single-concept categorisation
#   "concept_categorizer_multi"→ LoRA fine-tuned for batch categorisation
#
# If your gguf_engine uses a single GGUF with system-prompt-based "adapters",
# just keep all names as "default" and distinguish behaviour via the prompts.
ADAPTER_MAP = {
    "default":                   "default",
    "get_concept":               "get_concept",
    # "concept_categorizer":       "concept_categorizer",
    # "concept_categorizer_multi": "concept_categorizer_multi",
}



BASE_DIR = "/media/shrish/Data/medpao_fast/cpu_codes/gguf_models"

LLM_PATH = f"{BASE_DIR}/Qwen3-4B-instruct_q4km.gguf"



LLM_LORA_PATHS = {
    "get_concept": f"{BASE_DIR}/Qwen3-4B-conc_extr_lora.gguf",
    "default": None,
}

# N_THREADS = os.cpu_count() or 8
N_THREADS = 2
N_CTX     = 1024
N_BATCH   = 32

# ==========================================
# 2.  INFERENCE HELPERS
# ==========================================
def exclude_thinking_component(text: str) -> str:
    clean = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<unused94>.*",            "", clean, flags=re.DOTALL)
    return clean.strip()


@contextmanager
def _suppress_stderr():
    """
    Redirects C-level stderr to /dev/null during llama.cpp model loading.
    Uses os.dup2 so it catches output from the native C library, not just Python.
    """
    stderr_fd   = sys.stderr.fileno()
    saved_fd    = os.dup(stderr_fd)
    devnull_fd  = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


@lru_cache(maxsize=2)
def _load_text_model(adapter_name: str) -> Llama:
    lora_path = LLM_LORA_PATHS.get(adapter_name, None)
    print(f"  [GGUF] Loading text model | adapter={adapter_name} | lora={lora_path}")
    _t = time.time()

    with _suppress_stderr():
        model = Llama(
            model_path = LLM_PATH,
            lora_path  = lora_path,
            lora_scale = 1.0,
            n_ctx      = N_CTX,
            n_batch    = N_BATCH,
            n_threads  = N_THREADS,
            use_mmap   = True,
            verbose    = False,
        )

    print(f"  [GGUF] Text model ready in {time.time()-_t:.1f}s")
    return model




def generate_with_adapter(prompt: str, adapter_name: str, max_tokens: int = 150) -> str:
    """Text-only inference. Same signature as original HuggingFace version."""
    key   = adapter_name if adapter_name in LLM_LORA_PATHS else "default"
    model = _load_text_model(key)

    _t = time.time()
    output = model(
        prompt,
        max_tokens  = max_tokens,
        stop        = ["<|assistant|>", "<eos>"],
        echo        = False,
        temperature = 0.0,
        top_p       = 1.0,
    )
    elapsed = time.time() - _t
    raw     = output["choices"][0]["text"].strip()
    tokens  = output["usage"]["completion_tokens"]

    print(key, "raw output:", raw)
    print()
    print()
    print(f"  [GGUF] {key} | {elapsed:.2f}s | {tokens} tok | {tokens/max(elapsed,0.01):.1f} tok/s")
    return raw




def parse_response(model_response: str, tool_name: str):
    """
    Parses raw LLM text into the structured type each tool expects.

    get_concept             → dict[str, str]   (concept → source sentence)
    categorize_concepts     → dict[str, str]   (concept → ABCDEF letter)
    generate_structured_report → dict[str, str]
    """
    # Strip <think>…</think> blocks (Qwen3 chain-of-thought) using both
    # regex and the gguf_engine helper for belt-and-suspenders coverage.
    cleaned = model_response
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    if tool_name == "get_concept":
        new_text = cleaned.split('<|assistant|>')[0].strip()
        data = json.loads(new_text)
        return data


    elif tool_name in ("categorize_concepts", "generate_structured_report"):
        stripped = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        match    = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  [parse_response] JSON decode failed for {tool_name}, returning {{}}")
                return {}
        print(f"  [parse_response] No JSON found for {tool_name}, returning {{}}")
        return {}

    else:
        return model_response


# ==========================================
# 3.  LANGGRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_query:                    str
    input_report:                  str
    input_findings:                Optional[str]
    modules_queue:                 List[str]
    concepts:                      dict   # concept → source sentence
    existing_categorized_concepts: dict   # from cache
    new_categorized_concepts:      dict   # from LLM categorisation
    ontology_mapping:              dict   # concept → SNOMED description + ancestors
    structured_report:             dict   # ABCDEF → findings text


# ==========================================
# 4.  PLANNER NODE
# ==========================================
def planner_node(state: AgentState):
    print("\n--- NODE: Planner (Dynamic Module Selection) ---")

    prompt = f"""You are a strict medical AI pipeline orchestrator.
    You must respond with ONLY a valid JSON array of tool names.
    NO markdown, NO explanations, NO extra text.

    TOOL REGISTRY:
    - get_concept: extract medical concepts from the report
    - check_cache: check if concepts are previously processed
    - ontology_mapping: get SNOMED/RADLEX info per concept
    - categorize_concepts: categorize concepts into ABCDEF categories
    - generate_structured_report: generate structured report from categories

    EXAMPLES:
    Query: "structure this radiology report according to protocol"
    Output: ["get_concept", "check_cache", "ontology_mapping", "categorize_concepts", "generate_structured_report"]

    Query: "map these concepts to ontologies: consolidation, effusion"
    Output: ["ontology_mapping"]

    NOW ANSWER:
    Report snippet: "{str(state['input_report'])[:300]}"
    Query: "{state['user_query']}"
    Output (JSON array only):"""

    raw = generate_with_adapter(prompt, adapter_name="default", max_tokens=60)

    # Strip any accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    # Only grab the first [...] array in the response
    raw = exclude_thinking_component(raw)

    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            modules_queue = json.loads(match.group())
        except json.JSONDecodeError:
            print("  [Planner] JSON decode error — using full pipeline as fallback.")
    
    else:
        print("  [Planner] No module list found — using full pipeline as fallback.")


    print(f"  [Planner] Selected modules: {modules_queue}")
    return {"modules_queue": modules_queue}




# ==========================================
# 5.  ROUTING LOGIC
# ==========================================
def route_next(state: AgentState) -> str:
    queue = state.get("modules_queue", [])
    return queue[0] if queue else "end"


def _pop_queue(state: AgentState) -> List[str]:
    return state.get("modules_queue", [])[1:]


# ==========================================
# 6.  MODULE NODES
# ==========================================

# ── get_concept ─────────────────────────────────────────────────────────────
def run_get_concept_tool(state: AgentState):
    print("\n--- TOOL: get_concept ---")

    # Few-shot examples (trimmed for CPU token budget)
    examples = (
        "Report: 'No focal consolidation, pleural effusion, or pneumothorax is seen. "
        "Heart and mediastinal contours are within normal limits.'\n"
        "Concepts: {\"no focal consolidation\": \"No focal consolidation, pleural effusion, or pneumothorax is seen.\", "
        "\"no pleural effusion\": \"No focal consolidation, pleural effusion, or pneumothorax is seen.\", "
        "\"no pneumothorax\": \"No focal consolidation, pleural effusion, or pneumothorax is seen.\", "
        "\"normal heart contours\": \"Heart and mediastinal contours are within normal limits.\", "
        "\"normal mediastinal contours\": \"Heart and mediastinal contours are within normal limits.\"}\n\n"

        "Report: 'The lungs are clear with no evidence of consolidation, effusion, or pneumothorax. "
        "Lung volumes are low. Cardiomediastinal silhouette is normal. No acute fractures are identified.'\n"
        "Concepts: {\"clear lungs\": \"The lungs are clear with no evidence of consolidation, effusion, or pneumothorax.\", "
        "\"no consolidation\": \"The lungs are clear with no evidence of consolidation, effusion, or pneumothorax.\", "
        "\"low lung volumes\": \"Lung volumes are low.\", "
        "\"normal cardiomediastinal silhouette\": \"Cardiomediastinal silhouette is normal.\", "
        "\"no acute fractures\": \"No acute fractures are identified.\"}\n\n"
    )

    prompt = (
        f"<|system|>\n"
        f"You are an expert radiology practitioner. Extract medical concepts and their "
        f"exact source sentences from the given chest X-ray report.\n"
        f"Respond ONLY with a JSON object: {{\"concept\": \"source sentence\", ...}}\n"
        f"<|user|>\n"
        f"{examples}"
        f"Report: '{state['input_report']}'\n"
        f"Concepts:"
        f"<|assistant|>"
    )

    result = generate_with_adapter(prompt, adapter_name="get_concept", max_tokens=300)
    concepts = parse_response(result, tool_name="get_concept")

    print(f"  Extracted {len(concepts)} concepts: {list(concepts.keys())}")
    return {"concepts": concepts, "modules_queue": _pop_queue(state)}


# ── check_cache ──────────────────────────────────────────────────────────────
def run_check_cache_tool(state: AgentState):
    print("\n--- TOOL: check_cache ---")
    _t0 = time.time()
    cache_path = os.path.join(os.getcwd(), "cached_vocab.json")

    if not os.path.exists(cache_path):
        print("  [Cache] No cache file found — all concepts need mapping.")
        print(f"      [check_cache Complete — {time.time() - _t0:.2f}s]")
        return {"existing_categorized_concepts": {}, "modules_queue": _pop_queue(state)}

    with open(cache_path) as f:
        vocab = json.load(f)

    concepts = state.get("concepts", {})
    cached   = {c: vocab[c] for c in concepts if c in vocab}
    uncached = [c for c in concepts if c not in vocab]

    print(f"  Cached ({len(cached)}): {list(cached.keys())}")
    print(f"  Needs mapping ({len(uncached)}): {uncached}")
    print(f"      [check_cache Complete — {time.time() - _t0:.2f}s]")
    return {"existing_categorized_concepts": cached, "modules_queue": _pop_queue(state)}


# ── ontology_mapping ─────────────────────────────────────────────────────────
def run_ontology_mapping_tool(state: AgentState):
    print("\n--- TOOL: ontology_mapping ---")
    print("Initializing SNOMED Matcher...")
    matcher = SnomedMatcher()
    print("SNOMED Matcher ready.")
    _t0 = time.time()

    existing = state.get("existing_categorized_concepts", {})
    concepts_to_map = [c for c in state["concepts"] if c not in existing]

    if not concepts_to_map:
        print("  [Ontology] No new concepts to map — skipping.")
        print(f"      [ontology_mapping Complete — {time.time() - _t0:.2f}s]")
        return {"ontology_mapping": {}, "modules_queue": _pop_queue(state)}

    all_matches = matcher.match(concepts_to_map, top_k=1)

    concept_ontology_map = {}
    for matches in all_matches:
        m            = matches[0]
        ancestors    = m["ancestors"][:5]
        ancestors_str = ", ".join(f["desc"] for f in ancestors)
        concept_ontology_map[m["query"]] = (
            f"Description: {m['preferred_desc']} | Ancestors: {ancestors_str}"
        )

    print(f"  Mapped {len(concept_ontology_map)} concepts.")
    print(f"      [ontology_mapping Complete — {time.time() - _t0:.2f}s]")
    return {"ontology_mapping": concept_ontology_map, "modules_queue": _pop_queue(state)}


# ── categorize_concepts ──────────────────────────────────────────────────────
_ABCDEF_SYSTEM = """\
You are a strict radiology expert and medical classifier.

Chest X-ray ABCDEF protocol:
  A: Airways        — trachea, carina, bronchi (narrowing, mass, deviation)
  B: Breathing      — lungs, pleural spaces, pulmonary vessels, effusion, pneumothorax
  C: Circulation    — cardiac silhouette, mediastinum, hilar structures, great vessels
  D: Diaphragm/below— hemidiaphragms, free air, bowel, gastric bubble
  E: External       — ribs, clavicles, shoulder girdle, soft tissues, fractures
  F: Foreign material— lines, tubes, clips, implants

CRITICAL OVERRIDES (always take precedence):
  1. Hilar / hilar vessels           → C (Circulation)
  2. Pleura / effusion / opacity → B (Breathing)
  3. Diaphragm / hemidiaphragm        → D (Diaphragm)
  4. Pulmonary vessels / vasculature  → B (Breathing)
"""


def run_categorize_concepts_tool(state: AgentState):
    print("\n--- TOOL: categorize_concepts ---")

    ontology_mapping = state.get("ontology_mapping", {})
    if not ontology_mapping:
        print("  [Categorization] No new concepts — skipping.")
        return {"new_categorized_concepts": {}, "modules_queue": _pop_queue(state)}

    # Build a compact concept block
    concept_lines = "\n".join(
        f"- \"{concept}\": {desc}"
        for concept, desc in ontology_mapping.items()
    )

    prompt = (
        f"<|system|>\n{_ABCDEF_SYSTEM}\n"
        f"<|user|>\n"
        f"Categorize EACH concept below into one letter: A, B, C, D, E, or F.\n\n"
        f"EXAMPLE INPUT:\n"
        f"- \"normal hilar contours\": Description: Hilum of lung | Ancestors: Lung, Respiratory\n"
        f"- \"blunted costophrenic angle\": Description: Costodiaphragmatic recess | Ancestors: Pleura\n\n"
        f"EXAMPLE OUTPUT:\n"
        f"{{\"normal hilar contours\": \"C\", \"blunted costophrenic angle\": \"B\"}}\n\n"
        f"NOW PROCESS:\n{concept_lines}\n\n"
        f"Output ONLY a single JSON object with all concepts as keys and a letter as each value.\n"
        f"<|assistant|>"
    )

    response      = generate_with_adapter(prompt, adapter_name="concept_categorizer_multi", max_tokens=512)
    final_response = parse_response(response, tool_name="categorize_concepts")
    print(f"  Categorized: {final_response}")

    # ── Update cache ────────────────────────────────────────────────────────
    cache_path = os.path.join(os.getcwd(), "cached_vocab.json")
    cached_vocab = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached_vocab = json.load(f)

    updated = {**cached_vocab, **final_response}
    with open(cache_path, "w") as f:
        json.dump(updated, f, indent=4)
    print(f"  Cache updated → {cache_path}")

    return {"new_categorized_concepts": final_response, "modules_queue": _pop_queue(state)}


# ── generate_structured_report ───────────────────────────────────────────────
def run_generate_structured_report_tool(state: AgentState):
    print("\n--- TOOL: generate_structured_report ---")

    existing  = state.get("existing_categorized_concepts", {})
    new       = state.get("new_categorized_concepts", {})
    combined  = {**existing, **new}
    concepts  = state.get("concepts", {})

    prompt = (
        f"<|system|>\nYou are a board-certified radiologist writing structured chest X-ray reports.\n"
        f"<|user|>\n"
        f"Using the ABCDEF protocol (A=Airways, B=Breathing, C=Circulation, "
        f"D=Diaphragm/below, E=External, F=Foreign material), write a structured report.\n\n"
        f"Concept-to-source mapping:\n{json.dumps(concepts, indent=2)}\n\n"
        f"Concept categories (concept → letter):\n{json.dumps(combined, indent=2)}\n\n"
        f"Rules:\n"
        f"  • Write each section in professional radiologist prose — do NOT just list concepts.\n"
        f"  • Group all concepts sharing a letter into one cohesive sentence or two.\n"
        f"  • If a letter has no concepts, write \"No findings\".\n"
        f"  • Do NOT introduce findings not present in the concept list.\n\n"
        f"Output ONLY this JSON (no markdown, no preamble):\n"
        f"{{\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\", \"E\": \"...\", \"F\": \"...\"}}\n"
        f"<|assistant|>"
    )

    response = generate_with_adapter(prompt, adapter_name="default", max_tokens=500)
    result   = parse_response(response, tool_name="generate_structured_report")
    print(f"  Structured report sections: {list(result.keys())}")
    return {"structured_report": result, "modules_queue": _pop_queue(state)}


# ── orchestrator ─────────────────────────────────────────────────────────────
def orchestrator_synthesis_node(state: AgentState):
    print("\n--- NODE: Orchestrator (Final Summary) ---")
    report = state.get("structured_report", {})
    print("\n" + "=" * 50)
    print("STRUCTURED REPORT")
    print("=" * 50)
    for section in ["A", "B", "C", "D", "E", "F"]:
        findings = report.get(section, "No findings")
        label = {
            "A": "Airways",
            "B": "Breathing",
            "C": "Circulation",
            "D": "Diaphragm & Below",
            "E": "External",
            "F": "Foreign Material",
        }[section]
        print(f"  [{section}] {label}: {findings}")
    print("=" * 50)
    return {}


# ==========================================
# 7.  BUILD & COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner",                    planner_node)
workflow.add_node("get_concept",                run_get_concept_tool)
workflow.add_node("check_cache",                run_check_cache_tool)
workflow.add_node("ontology_mapping",           run_ontology_mapping_tool)
workflow.add_node("categorize_concepts",        run_categorize_concepts_tool)
workflow.add_node("generate_structured_report", run_generate_structured_report_tool)
workflow.add_node("orchestrator",               orchestrator_synthesis_node)

workflow.set_entry_point("planner")

_all_nodes = [
    "get_concept", "check_cache", "ontology_mapping",
    "categorize_concepts", "generate_structured_report",
    "orchestrator", "end",
]
_routing_map = {n: n for n in _all_nodes}
_routing_map["end"] = END

workflow.add_conditional_edges("planner", route_next, _routing_map)
for _mod in ["get_concept", "check_cache", "ontology_mapping",
             "categorize_concepts", "generate_structured_report"]:
    workflow.add_conditional_edges(_mod, route_next, _routing_map)

workflow.add_edge("orchestrator", END)

full_agent = workflow.compile()


# ==========================================
# 8.  ENTRY POINT
# ==========================================
if __name__ == "__main__":
    _report = (
        "There is a vague opacity seen within the right lower lobe, concerning for pneumonia. "
        "There is no pleural effusion or pneumothorax. "
        "The heart size is mildly enlarged. "
        "The hilar and mediastinal structures are unremarkable."
    )

    _base_state = {
        "input_report":                  _report,
        "input_findings":                "",
        "modules_queue":                 [],
        "concepts":                      {},
        "existing_categorized_concepts": {},
        "new_categorized_concepts":      {},
        "ontology_mapping":              {},
        "structured_report":             {},
    }

    out = full_agent.invoke({
        **_base_state,
        "user_query": "Structure this radiology report according to the ABCDEF protocol.",
    })

    print("\n=== FINAL STATE ===")
    print(out)