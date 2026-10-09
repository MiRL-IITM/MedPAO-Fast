"""
MedPAO-Fast — Gradio Interface
Run: python gradio_ui.py

Requires langgraph_agent.py (with the compiled `full_agent` graph) and
ontoology_mapping.py in the same directory. Model paths, base-model ID,
LoRA repo/subfolders, device, and SNOMED index directory used by
langgraph_agent.py are configured via ITS OWN CLI flags — see
`python langgraph_agent.py --help`, or the repo README. Any such flags
passed to this script are forwarded through (langgraph_agent.py uses
`argparse.parse_known_args()`, so unrecognised flags are ignored rather
than causing an error).

This file's own CLI flags configure the Gradio server itself (host, port,
sharing) and the optional HF_HOME cache directory.

Traceability  ("Linked view")
-----------------------------
The structured report is shown side-by-side with the source report. Every
generated sentence is fuzzy-matched (rapidfuzz) to the concept it came from,
and that concept's source sentence is located in the input report. Hovering a
sentence on EITHER side lights up its counterpart on the other side (same
colour, same link number) and dims everything else. Hovering a structured
sentence also opens an inline drawer with concept, SNOMED CT mapping,
category and match confidence. Pure CSS (:has) — no extra model calls, no JS.
"""
import argparse
import html
import os
import re
import time


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedPAO-Fast Gradio interface."
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default="/media/shrish/Data/huggingface_models",
        help="HF_HOME cache directory. Set before any HuggingFace-dependent import.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host/interface to bind the Gradio app to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port for the Gradio app.",
    )
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a public Gradio share link. Use --no-share to disable.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MedPAO-Fast",
        help="Browser tab / header title for the app.",
    )
    parser.add_argument(
        "--sample-report",
        type=str,
        default=(
            "PA and lateral views of the chest are submitted. Lungs appear well inflated "
            "without evidence of focal airspace consolidation, pleural effusions, pulmonary "
            "edema, or pneumothorax. Cardiac and mediastinal contours are within normal limits. "
            "No acute bony abnormality is appreciated."
        ),
        help="Sample report text shown via the 'Load example report' button.",
    )
    parser.add_argument(
        "--trace-score-threshold",
        type=float,
        default=55.0,
        help="Minimum rapidfuzz match score (0-100) for a report sentence to be "
             "linked to a concept in the traceability view. Lower = more "
             "(possibly weaker) matches shown; higher = fewer, more confident ones.",
    )
    return parser


_parser = build_arg_parser()
args, _unknown = _parser.parse_known_args()

os.environ['HF_HOME'] = args.hf_home

import gradio as gr
import json
import time
import traceback
from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------------
# Import the compiled LangGraph agent.
# ---------------------------------------------------------------------------
from langgraph_agent import full_agent


# ---------------------------------------------------------------------------
# Section + pipeline metadata
# ---------------------------------------------------------------------------
SECTION_META = {
    "P": ("Projection",  "📐", "#4A9EFF"),
    "A": ("Airways",     "🫧", "#A78BFA"),
    "B": ("Breathing",   "🫁", "#34D399"),
    "C": ("Circulation", "❤️",  "#F87171"),
    "D": ("Diaphragm",   "⬇️",  "#FBBF24"),
    "E": ("External",    "🦴", "#FB923C"),
    "F": ("Foreign",     "🔩", "#60A5FA"),
}

PIPELINE_STEPS = [
    ("get_concept",                "01", "Concept Extraction"),
    ("check_cache",                "02", "Cache Lookup"),
    ("ontology_mapping",           "03", "Ontology Mapping"),
    ("categorize_concepts",        "04", "ABCDEF Classification"),
    ("generate_structured_report", "05", "Report Generation"),
]

STEP_INDEX = {key: i for i, (key, _, _) in enumerate(PIPELINE_STEPS)}

STEP_LABEL = {
    "get_concept":                "Extracting concepts…",
    "check_cache":                "Checking cache…",
    "ontology_mapping":           "Mapping ontologies…",
    "categorize_concepts":        "Classifying ABCDEF…",
    "generate_structured_report": "Generating report…",
}

# Each source sentence gets its own link colour + number. CSS rules are
# pre-generated for this many source sentences (more than enough for a CXR).
MAX_LINKS = 40
LINK_COLORS = ["#38BDF8", "#F472B6", "#34D399", "#FBBF24",
               "#A78BFA", "#FB923C", "#2DD4BF", "#F87171"]


def _link_color(i: int) -> str:
    return LINK_COLORS[i % len(LINK_COLORS)]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _esc(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s), quote=True)


def _fmt(s: float) -> str:
    return f"{s:.2f}s"


def _split_sentences(text: str) -> list[str]:
    """Naive but dependency-free sentence splitter."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------
def _locate_source(source: str, src_sents: list[str]) -> list[int]:
    """Indices of the input-report sentences that a concept's source text
    covers. Exact containment first (handles multi-sentence sources), fuzzy
    partial match as a fallback."""
    if not source or not src_sents:
        return []
    ns = _norm(source)
    exact = [i for i, s in enumerate(src_sents)
             if _norm(s) and (_norm(s) in ns or ns in _norm(s))]
    if exact:
        return [i for i in exact if i < MAX_LINKS]
    res = process.extractOne(ns, [_norm(s) for s in src_sents], scorer=fuzz.partial_ratio)
    if res is not None and res[1] >= 70 and res[2] < MAX_LINKS:
        return [res[2]]
    return []


def build_trace_map(
    concepts: dict,
    ontology_mapping: dict,
    combined_categories: dict,
    structured_report: dict,
    input_report: str = "",
    score_threshold: float = 55.0,
) -> dict:
    """
    Returns {section: [{"sentence": str, "trace": dict|None}, ...]}
    `trace`, when present, has: concept, source, ontology, category, score,
    src_idx (indices of linked sentences in the input report).
    """
    src_sents = _split_sentences(input_report)
    trace_map = {}
    for section, text in structured_report.items():
        section_concepts = [c for c, cat in combined_categories.items() if cat == section]
        rows = []
        for sent in _split_sentences(text):
            match_info = None
            if section_concepts:
                result = process.extractOne(sent, section_concepts, scorer=fuzz.token_set_ratio)
                if result is not None:
                    best_concept, score, _ = result
                    if score >= score_threshold:
                        source = concepts.get(best_concept, "")
                        match_info = {
                            "concept":  best_concept,
                            "source":   source,
                            "ontology": ontology_mapping.get(
                                best_concept,
                                "Cached concept — ontology wasn't recomputed this run.",
                            ),
                            "category": combined_categories.get(best_concept, section),
                            "score":    round(float(score), 1),
                            "src_idx":  _locate_source(source, src_sents),
                        }
            rows.append({"sentence": sent, "trace": match_info})
        trace_map[section] = rows
    return trace_map


def _trace_for(acc: dict, report: dict, threshold: float) -> dict:
    if not report:
        return {}
    combined_categories = {
        **acc.get("existing_categorized_concepts", {}),
        **acc.get("new_categorized_concepts", {}),
    }
    return build_trace_map(
        concepts=acc.get("concepts", {}),
        ontology_mapping=acc.get("ontology_mapping", {}),
        combined_categories=combined_categories,
        structured_report=report,
        input_report=acc.get("input_report", ""),
        score_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def _pipeline_html(
    active_step: int = -1,
    done: bool = False,
    step_times: dict = None,
    total_time: float = None,
) -> str:
    step_times = step_times or {}
    n = len(PIPELINE_STEPS)
    completed = n if done else max(0, active_step)
    percent = int(round((completed / n) * 100)) if n else 0

    items = ""
    for i, (key, num, label) in enumerate(PIPELINE_STEPS):
        if done or i < active_step:
            state_cls, icon = "step-done", "✓"
        elif i == active_step:
            state_cls, icon = "step-active", num
        else:
            state_cls, icon = "step-pending", num

        t = step_times.get(key)
        time_badge = (
            f'<div class="step-time">{_fmt(t)}</div>'
            if t is not None and (done or i < active_step)
            else '<div class="step-time">&nbsp;</div>'
        )
        connector_done = "connector-done" if (done or i < active_step) else ""
        connector = f'<div class="connector {connector_done}"></div>' if i < n - 1 else ""
        items += f"""
        <div class="pipeline-item">
            <div class="step-bubble {state_cls}">{icon}</div>
            <div class="step-label {state_cls}">{label}</div>
            {time_badge}
        </div>{connector}"""

    progress_bar = (
        f'<div class="pipeline-progress">'
        f'<div class="pipeline-progress-fill" style="width:{percent}%;"></div>'
        f'</div>'
    )

    total_row = ""
    if total_time is not None:
        total_row = f"""
        <div class="pipeline-total">
            <span class="total-label">⏱ Total pipeline time</span>
            <span class="total-value">{_fmt(total_time)}</span>
        </div>"""

    return (
        f'<div class="pipeline-wrap">{progress_bar}'
        f'<div class="pipeline-track">{items}</div>{total_row}</div>'
    )


def _concepts_html(concepts: dict) -> str:
    if not concepts:
        return '<p class="empty-msg">No concepts extracted yet.</p>'
    rows = "".join(f"""
        <div class="concept-row">
            <span class="concept-tag">{_esc(c)}</span>
            <span class="concept-src">↳ {_esc(s)}</span>
        </div>""" for c, s in concepts.items())
    return f'<div class="concepts-grid">{rows}</div>'


def _badges(idxs: list[int]) -> str:
    return "".join(
        f'<sup class="lk" style="--lc:{_link_color(i)};">{i + 1}</sup>' for i in idxs
    )


def _studio_html(source_text: str, report: dict, trace: dict = None) -> str:
    """Side-by-side linked view: source report (left) ↔ structured report (right)."""
    trace = trace or {}
    src_sents = _split_sentences(source_text)

    # which source sentences are referenced by at least one output sentence
    linked_src = set()
    n_out = n_linked = 0
    for rows in trace.values():
        for r in rows:
            n_out += 1
            if r["trace"]:
                n_linked += 1
                linked_src.update(r["trace"].get("src_idx", []))

    # ── left: source panel ────────────────────────────────────────────────
    if src_sents:
        parts = []
        for i, s in enumerate(src_sents):
            if i in linked_src:
                parts.append(
                    f'<span class="ss l{i}" style="--lc:{_link_color(i)};">'
                    f'<sup class="lk">{i + 1}</sup>{_esc(s)}</span>'
                )
            else:
                parts.append(f'<span class="su">{_esc(s)}</span>')
        src_body = " ".join(parts)
    else:
        src_body = '<span class="empty-inline">The input report appears here once you run the pipeline.</span>'

    # ── right: structured cards ───────────────────────────────────────────
    if not report:
        out_body = '<p class="empty-msg">Structured report not generated yet.</p>'
    else:
        cards = ""
        for key, text in report.items():
            label, icon, color = SECTION_META.get(key, (key, "📋", "#94A3B8"))
            no_finding = text.strip().lower() in ("no findings.", "no findings", "no findings identified.")
            dim = " dim" if no_finding else ""

            rows = trace.get(key)
            if rows:
                pieces = []
                for row in rows:
                    sent = _esc(row["sentence"])
                    t = row["trace"]
                    if not t:
                        pieces.append(f'<div class="tu">{sent}</div>')
                        continue
                    idxs = t.get("src_idx", [])
                    cls = " ".join(f"l{i}" for i in idxs)
                    lc = _link_color(idxs[0]) if idxs else color
                    cat_letter = t["category"]
                    cat_label = SECTION_META.get(cat_letter, (cat_letter, "", ""))[0]
                    pieces.append(f"""
                    <div class="ts {cls}" style="--lc:{lc};">
                        <div class="ts-main">{_badges(idxs)}<span class="ts-text">{sent}</span></div>
                        <div class="ts-drawer"><div class="ts-drawer-in">
                            <div class="dr-row"><span class="dr-k">Concept</span><span class="dr-v dr-concept">{_esc(t['concept'])}</span></div>
                            <div class="dr-row"><span class="dr-k">SNOMED CT</span><span class="dr-v dr-onto">{_esc(t['ontology']) or '—'}</span></div>
                            <div class="dr-foot">
                                <span class="dr-pill" style="color:{color};border-color:{color}66;">{_esc(cat_letter)} · {_esc(cat_label)}</span>
                                <span class="dr-conf"><span class="dr-bar"><span style="width:{min(100, t['score'])}%;"></span></span>{t['score']}%</span>
                            </div>
                        </div></div>
                    </div>""")
                body = "".join(pieces)
            else:
                body = f'<div class="tu">{_esc(text)}</div>'

            cards += f"""
            <div class="report-card{dim}">
                <div class="card-header" style="border-left:3px solid {color};">
                    <span class="card-icon">{icon}</span>
                    <span class="card-key" style="color:{color};">{key}</span>
                    <span class="card-label">{label}</span>
                </div>
                <div class="card-text">{body}</div>
            </div>"""
        out_body = f'<div class="report-grid">{cards}</div>'

    stats = ""
    if report:
        stats = f"""
        <div class="studio-stats">
            <div class="stat"><b>{n_out}</b><span>Output sentences</span></div>
            <div class="stat"><b>{n_linked}</b><span>Source-linked</span></div>
            <div class="stat"><b>{len(linked_src)}</b><span>Source spans used</span></div>
        </div>"""

    return f"""
    <div class="studio">
        {stats}
        <div class="studio-cols">
            <div class="studio-panel">
                <div class="studio-title"><span class="studio-num">1</span>Source report</div>
                <div class="src-body">{src_body}</div>
            </div>
            <div class="studio-link" aria-hidden="true"><span></span></div>
            <div class="studio-panel">
                <div class="studio-title"><span class="studio-num">2</span>Structured report</div>
                <div class="out-body">{out_body}</div>
            </div>
        </div>
    </div>"""


def _ontology_html(onto: dict) -> str:
    if not onto:
        return '<p class="empty-msg">No ontology data.</p>'
    rows = "".join(f"""
        <div class="onto-row">
            <div class="onto-concept">{_esc(c)}</div>
            <div class="onto-info">{_esc(info)}</div>
        </div>""" for c, info in onto.items())
    return f'<div class="onto-list">{rows}</div>'


def _categories_html(cats: dict) -> str:
    if not cats:
        return '<p class="empty-msg">No categories yet.</p>'
    buckets: dict[str, list] = {}
    for concept, cat in cats.items():
        buckets.setdefault(cat, []).append(concept)
    rows = ""
    for cat in sorted(buckets):
        label, icon, color = SECTION_META.get(cat, (cat, "📋", "#94A3B8"))
        chips = "".join(
            f'<span class="cat-chip" style="border-color:{color};color:{color};">{_esc(c)}</span>'
            for c in buckets[cat]
        )
        rows += f"""
        <div class="cat-row">
            <div class="cat-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">
                {icon} {_esc(cat)} · {_esc(label)}
            </div>
            <div class="cat-chips">{chips}</div>
        </div>"""
    return f'<div class="cat-list">{rows}</div>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def _link_css() -> str:
    """Per-source-sentence hover rules (class l{i} is shared by the source span
    and every output sentence that came from it). Driven by :has()."""
    rules = []
    for i in range(MAX_LINKS):
        rules.append(f"""
.studio:has(.ts.l{i}:hover) .ss.l{i},
.studio:has(.ss.l{i}:hover) .ss.l{i} {{
    background:color-mix(in srgb, var(--lc) 30%, transparent);
    box-shadow:0 0 0 1px var(--lc), 0 0 22px color-mix(in srgb, var(--lc) 45%, transparent);
    color:#fff !important;
}}
.studio:has(.ts.l{i}:hover) .ss.l{i} .lk,
.studio:has(.ss.l{i}:hover) .ss.l{i} .lk {{ background:var(--lc); color:#06101c !important; }}
.studio:has(.ss.l{i}:hover) .ts.l{i},
.studio:has(.ts.l{i}:hover) .ts.l{i} {{
    background:color-mix(in srgb, var(--lc) 16%, transparent);
    border-color:var(--lc);
    box-shadow:0 0 0 1px color-mix(in srgb, var(--lc) 50%, transparent);
}}
.studio:has(.ss.l{i}:hover) .ts.l{i} .lk {{ background:var(--lc); color:#06101c !important; }}
.studio:has(.ts.l{i}:hover) .ss:not(.l{i}),
.studio:has(.ss.l{i}:hover) .ss:not(.l{i}),
.studio:has(.ts.l{i}:hover) .su,
.studio:has(.ss.l{i}:hover) .su {{ opacity:.28; }}
.studio:has(.ss.l{i}:hover) .ts:not(.l{i}),
.studio:has(.ss.l{i}:hover) .tu {{ opacity:.3; }}
.studio:has(.ts.l{i}:hover) .ts:not(.l{i}),
.studio:has(.ts.l{i}:hover) .tu {{ opacity:.3; }}""")
    return "\n".join(rules)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0B0F1A;
    --surface:   #111827;
    --surface2:  #1A2235;
    --surface3:  #202B45;
    --border:    #243554;
    --text:      #E2E8F0;
    --text-mid:  #94A3B8;
    --accent:    #38BDF8;
    --accent2:   #F59E0B;
    --danger:    #F87171;
    --success:   #34D399;
    --fh: 'DM Serif Display', serif;
    --fb: 'Sora', sans-serif;
    --fm: 'IBM Plex Mono', monospace;
    --r: 10px;
    --r-lg: 16px;
    --shadow: 0 8px 28px rgba(0,0,0,.35);
}

* { box-sizing:border-box; margin:0; padding:0; }

body, .gradio-container, .gradio-container *,
.svelte-1ipelgc, .prose, fieldset, legend,
.wrap, .label-wrap, .output-class {
    background-color: transparent;
    color: var(--text) !important;
    font-family: var(--fb) !important;
}
body, .gradio-container {
    background:
        radial-gradient(circle at 15% -10%, #16213a 0%, transparent 45%),
        radial-gradient(circle at 100% 0%, #14203a 0%, transparent 35%),
        var(--bg) !important;
}

.app-header {
    display:flex; align-items:center; gap:18px;
    margin-bottom:28px; padding-bottom:22px;
    position:relative;
}
.app-header::after {
    content:''; position:absolute; left:0; right:0; bottom:0; height:1px;
    background:linear-gradient(90deg, var(--accent) 0%, var(--border) 35%, transparent 100%);
}
.header-logo {
    width:54px; height:54px;
    background:linear-gradient(135deg,#0EA5E9,#6366F1);
    border-radius:16px; display:flex; align-items:center; justify-content:center;
    font-size:27px; box-shadow:0 0 32px #0EA5E950, inset 0 1px 0 rgba(255,255,255,.15);
    flex-shrink:0;
}
.header-text h1 {
    font-family:var(--fh) !important; font-size:2.1rem;
    color:#F8FAFC !important; letter-spacing:-0.5px; line-height:1;
    text-shadow: 0 2px 18px rgba(56,189,248,.25);
}
.header-text p {
    font-size:0.74rem; color:var(--text-mid) !important;
    margin-top:6px; font-family:var(--fm) !important; letter-spacing:0.06em;
}
.header-badge {
    margin-left:auto; padding:7px 16px;
    border:1px solid #0EA5E950; border-radius:20px;
    font-size:0.72rem; font-family:var(--fm) !important;
    color:var(--accent) !important;
    background:linear-gradient(135deg,#0EA5E914,#6366F114);
    letter-spacing:0.08em; white-space:nowrap;
}

textarea, input[type="text"] {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:var(--r) !important;
    font-family:var(--fm) !important;
    font-size:0.82rem !important;
}
textarea::placeholder { color:var(--text-mid) !important; opacity:1; }
textarea:focus { border-color:var(--accent) !important; outline:none !important; box-shadow:0 0 0 3px #38BDF820 !important; }

button[variant="primary"], .gr-button-primary {
    background:linear-gradient(135deg,#0284C7,#6366F1) !important;
    border:none !important; color:#fff !important;
    font-family:var(--fb) !important; font-weight:600 !important;
    font-size:0.9rem !important; padding:12px 32px !important;
    border-radius:var(--r) !important;
    box-shadow:0 4px 20px #6366F140 !important;
    transition:opacity .2s,transform .15s,box-shadow .2s !important;
}
button[variant="primary"]:hover {
    opacity:.92 !important; transform:translateY(-2px) !important;
    box-shadow:0 8px 26px #6366F160 !important;
}
button[variant="secondary"], .gr-button-secondary {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    font-family:var(--fb) !important;
    border-radius:var(--r) !important;
    font-size:0.85rem !important; padding:10px 20px !important;
    transition:border-color .2s, color .2s, transform .15s !important;
}
button[variant="secondary"]:hover {
    border-color:var(--accent) !important; color:var(--accent) !important;
    transform:translateY(-1px) !important;
}

.cm-editor, .cm-content, .cm-line, .gr-code {
    background:var(--surface2) !important;
    color:#A5D6FF !important;
    font-family:var(--fm) !important; font-size:0.78rem !important;
    border:1px solid var(--border) !important;
}

button[role="tab"] {
    font-family:var(--fm) !important; font-size:0.78rem !important;
    color:var(--text-mid) !important; background:transparent !important;
    border:none !important; border-bottom:2px solid transparent !important;
    padding:8px 16px !important; letter-spacing:.04em !important;
    transition: color .15s, border-color .15s !important;
}
button[role="tab"][aria-selected="true"] {
    color:var(--accent) !important; border-bottom-color:var(--accent) !important;
}

.section-label {
    font-family:var(--fm); font-size:.68rem; letter-spacing:.14em;
    text-transform:uppercase; color:var(--text-mid) !important;
    margin-bottom:10px; display:flex; align-items:center; gap:8px;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

.panel-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--r); padding:18px 20px; margin-bottom:14px;
    font-family:var(--fm); font-size:.77rem;
    color:var(--text) !important; line-height:1.65;
}

/* ── Pipeline ────────────────────────────────────────────────────────── */
.pipeline-wrap {
    background:linear-gradient(180deg, var(--surface) 0%, #0E1524 100%);
    border:1px solid var(--border);
    border-radius:var(--r-lg); margin-bottom:14px; overflow:hidden;
    box-shadow:var(--shadow);
}
.pipeline-progress { height:3px; background:var(--surface2); }
.pipeline-progress-fill {
    height:100%; background:linear-gradient(90deg, var(--accent), var(--success));
    transition:width .4s ease;
}
.pipeline-track {
    display:flex; align-items:flex-start;
    padding:22px 22px 14px; overflow-x:auto; gap:0;
}
.pipeline-item { display:flex; flex-direction:column; align-items:center; gap:6px; min-width:110px; }
.connector { flex:1; height:1px; background:var(--border); min-width:10px; margin-bottom:44px; margin-top:18px; transition:background .3s, height .3s; }
.connector.connector-done { background:linear-gradient(90deg, var(--success), var(--accent)); height:2px; }

.step-bubble {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--fm); font-size:.72rem; font-weight:500;
    transition:all .3s; flex-shrink:0;
}
.step-pending { background:var(--surface2); color:var(--text-mid) !important; border:1px solid var(--border); }
.step-active  { background:#0EA5E920; color:var(--accent) !important; border:1px solid var(--accent); box-shadow:0 0 16px #38BDF850; animation:pulse 1.4s infinite; }
.step-done    { background:#10B98120; color:var(--success) !important; border:1px solid var(--success); }
@keyframes pulse { 0%,100%{box-shadow:0 0 8px #38BDF840} 50%{box-shadow:0 0 24px #38BDF890} }

.step-label { font-size:.65rem; font-family:var(--fm); text-align:center; line-height:1.3; }
.step-label.step-pending { color:var(--text-mid) !important; }
.step-label.step-active  { color:var(--accent)    !important; font-weight:600; }
.step-label.step-done    { color:var(--success)   !important; }

.step-time {
    font-family:var(--fm); font-size:.62rem;
    color:var(--accent2) !important; text-align:center; min-height:14px;
}

.pipeline-total {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 22px; border-top:1px solid var(--border); background:#0EA5E908;
}
.total-label { font-family:var(--fm); font-size:.72rem; color:var(--text-mid) !important; letter-spacing:.06em; }
.total-value { font-family:var(--fm); font-size:.92rem; font-weight:600; color:var(--accent2) !important; letter-spacing:.04em; }

.status-bar {
    padding:11px 18px; border-radius:var(--r);
    font-family:var(--fm); font-size:.78rem;
    margin-bottom:20px; display:flex; align-items:center; gap:9px;
    border:1px solid transparent;
    transition:background .2s, border-color .2s, color .2s;
}
.status-idle    { background:#1E293B; color:var(--text-mid) !important; border-color:var(--border); }
.status-running { background:#0EA5E915; color:var(--accent) !important; border-color:#0EA5E930; }
.status-done    { background:#10B98115; color:var(--success) !important; border-color:#10B98130; }
.status-error   { background:#F8717115; color:var(--danger)  !important; border-color:#F8717130; }
.status-bar strong { color:var(--accent2) !important; }
.status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; flex-shrink:0; }
.status-running .status-dot { animation: blink 1.1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Hint banner ──────────────────────────────────────────────────────── */
.hint-banner {
    display:flex; align-items:flex-start; gap:10px;
    background:linear-gradient(135deg,#0EA5E912,#6366F112);
    border:1px dashed #0EA5E955;
    border-radius:var(--r); padding:11px 15px; margin-bottom:14px;
    font-size:.78rem; color:var(--text-mid) !important; line-height:1.55;
}
.hint-banner .hint-icon { font-size:1rem; flex-shrink:0; }
.hint-banner b { color:var(--accent) !important; font-weight:600; }

.concepts-grid { display:flex; flex-direction:column; gap:8px; }
.concept-row {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; padding:10px 14px;
    display:flex; flex-direction:column; gap:5px;
    transition:border-color .2s;
}
.concept-row:hover { border-color:#2A4A7F; }
.concept-tag { font-family:var(--fm); font-size:.8rem; color:var(--accent) !important; font-weight:500; }
.concept-src { font-size:.77rem; color:var(--text) !important; line-height:1.5; }

/* ══ LINKED STUDIO (source ↔ structured) ═════════════════════════════════ */
.studio {
    background:linear-gradient(180deg, var(--surface) 0%, #0E1524 100%);
    border:1px solid var(--border); border-radius:var(--r-lg);
    padding:18px; box-shadow:var(--shadow);
}
.studio-stats { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
.stat {
    flex:1; min-width:130px; display:flex; align-items:baseline; gap:9px;
    padding:10px 14px; border:1px solid var(--border); border-radius:var(--r);
    background:var(--surface2);
}
.stat b { font-family:var(--fh) !important; font-size:1.5rem; font-weight:400; color:#F8FAFC !important; }
.stat span { font-size:.68rem; color:var(--text-mid) !important; font-family:var(--fm); }

.studio-cols { display:flex; align-items:stretch; gap:0; }
.studio-panel {
    flex:1 1 0; min-width:0;
    background:#0A1120; border:1px solid var(--border);
    border-radius:14px; padding:16px 18px 18px;
}
.studio-link { width:34px; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.studio-link span {
    width:100%; height:2px; border-radius:2px;
    background:repeating-linear-gradient(90deg, var(--accent) 0 5px, transparent 5px 9px);
    opacity:.55;
}
.studio-title {
    display:flex; align-items:center; gap:9px; margin-bottom:14px;
    font-size:.84rem; font-weight:600; color:#F8FAFC !important;
}
.studio-num {
    width:22px; height:22px; border-radius:50%; display:inline-flex;
    align-items:center; justify-content:center;
    font-family:var(--fm); font-size:.68rem; font-weight:500;
    color:var(--accent) !important; border:1px solid var(--accent); background:#38BDF812;
}

/* source side */
.src-body {
    font-family:var(--fm); font-size:.8rem; line-height:2.15;
    color:var(--text) !important; max-height:620px; overflow-y:auto; padding-right:6px;
}
.empty-inline { color:var(--text-mid) !important; font-family:var(--fm); font-size:.78rem; }
.ss, .su { transition:opacity .2s, background .2s, box-shadow .2s; }
.ss {
    --lc:#38BDF8;
    position:relative; cursor:pointer; padding:3px 6px; margin:0 1px;
    border-radius:6px;
    background:color-mix(in srgb, var(--lc) 9%, transparent);
    box-decoration-break:clone; -webkit-box-decoration-break:clone;
    border-bottom:2px solid color-mix(in srgb, var(--lc) 55%, transparent);
}
.lk {
    display:inline-flex; align-items:center; justify-content:center;
    min-width:15px; height:15px; padding:0 4px; margin-right:6px;
    border-radius:8px; vertical-align:middle; position:relative; top:-1px;
    font-family:var(--fm); font-size:.6rem; font-weight:600; line-height:1;
    background:color-mix(in srgb, var(--lc) 25%, transparent);
    color:var(--lc) !important;
    border:1px solid color-mix(in srgb, var(--lc) 60%, transparent);
    transition:background .2s, color .2s;
}

/* output side */
.report-grid { display:flex; flex-direction:column; gap:12px; }
.report-card {
    background:linear-gradient(180deg, var(--surface2) 0%, #17203450 100%);
    border:1px solid var(--border);
    border-radius:12px; padding:14px 16px;
}
.report-card.dim { opacity:.4; }
.card-header { display:flex; align-items:center; gap:10px; padding-left:12px; margin-bottom:10px; }
.card-icon   { font-size:1.05rem; }
.card-key    { font-family:var(--fm); font-weight:700; font-size:.98rem; }
.card-label  { font-size:.69rem; color:var(--text-mid) !important; font-family:var(--fm); letter-spacing:.04em; margin-left:4px; }
.card-text   { display:flex; flex-direction:column; gap:6px; }

.tu { font-size:.84rem; line-height:1.65; padding:5px 10px 5px 12px; transition:opacity .2s; }

.ts {
    --lc:#38BDF8;
    position:relative; cursor:pointer;
    border:1px solid transparent; border-left:3px solid var(--lc);
    border-radius:9px; padding:7px 10px 7px 11px;
    background:color-mix(in srgb, var(--lc) 5%, transparent);
    transition:opacity .2s, background .2s, border-color .2s, box-shadow .2s;
}
.ts-main { font-size:.84rem; line-height:1.65; }
.ts-text { color:var(--text) !important; }

/* inline drawer — opens under the hovered sentence, never clipped */
.ts-drawer {
    display:grid; grid-template-rows:0fr; opacity:0;
    transition:grid-template-rows .22s ease, opacity .18s ease, margin-top .22s ease;
}
.ts-drawer-in { overflow:hidden; min-height:0; }
.ts:hover .ts-drawer { grid-template-rows:1fr; opacity:1; margin-top:9px; }
.ts-drawer-in {
    border-top:1px dashed color-mix(in srgb, var(--lc) 45%, transparent);
    padding-top:0;
}
.ts:hover .ts-drawer-in { padding-top:9px; }
.dr-row { display:flex; gap:12px; margin-bottom:7px; align-items:baseline; }
.dr-k {
    flex:0 0 78px; font-family:var(--fm); font-size:.64rem;
    color:var(--text-mid) !important; letter-spacing:.04em;
}
.dr-v { font-size:.78rem; line-height:1.5; color:var(--text) !important; white-space:pre-wrap; }
.dr-concept { font-family:var(--fm); color:var(--lc) !important; font-weight:500; }
.dr-onto { font-family:var(--fm); font-size:.73rem; color:#CBD5E1 !important; }
.dr-foot { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:4px; }
.dr-pill {
    font-family:var(--fm); font-size:.66rem; padding:2px 10px;
    border:1px solid; border-radius:20px;
}
.dr-conf { display:flex; align-items:center; gap:8px; font-family:var(--fm); font-size:.66rem; color:var(--text-mid) !important; }
.dr-bar { width:70px; height:4px; border-radius:3px; background:var(--surface3); overflow:hidden; display:inline-block; }
.dr-bar span { display:block; height:100%; border-radius:3px; background:linear-gradient(90deg, var(--accent), var(--success)); }

@media (max-width: 900px) {
    .studio-cols { flex-direction:column; }
    .studio-link { width:100%; height:26px; }
    .studio-link span { width:2px; height:100%;
        background:repeating-linear-gradient(180deg, var(--accent) 0 5px, transparent 5px 9px); }
}
@media (prefers-reduced-motion: reduce) {
    .ss, .su, .ts, .tu, .ts-drawer { transition:none !important; }
}

.onto-list { display:flex; flex-direction:column; gap:10px; }
.onto-row  { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; transition:border-color .2s; }
.onto-row:hover { border-color:#2A4A7F; }
.onto-concept { font-family:var(--fm); font-size:.8rem; color:var(--accent2) !important; margin-bottom:6px; font-weight:500; }
.onto-info    { font-size:.78rem; color:var(--text) !important; white-space:pre-wrap; line-height:1.55; }

.cat-list { display:flex; flex-direction:column; gap:10px; }
.cat-row  { display:flex; align-items:flex-start; gap:12px; flex-wrap:wrap; }
.cat-badge { padding:4px 12px; border-radius:20px; font-family:var(--fm); font-size:.73rem; white-space:nowrap; flex-shrink:0; }
.cat-chips { display:flex; flex-wrap:wrap; gap:6px; }
.cat-chip  { padding:3px 10px; border-radius:6px; border:1px solid; font-size:.72rem; font-family:var(--fm); background:transparent; }

.full-width-divider {
    border-top: 1px solid var(--border);
    margin-top: 28px;
    padding-top: 24px;
}

.empty-msg {
    color:var(--text-mid) !important; font-family:var(--fm); font-size:.8rem;
    padding:20px; text-align:center;
    border:1px dashed var(--border); border-radius:10px;
}

.app-footer {
    margin-top:40px; padding-top:20px; border-top:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between;
    font-family:var(--fm); font-size:.7rem; color:var(--text-mid) !important;
    flex-wrap:wrap; gap:10px;
}
.app-footer a { color:var(--accent) !important; text-decoration:none; }
.app-footer a:hover { text-decoration:underline; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
""" + _link_css()


# ---------------------------------------------------------------------------
# Processing function
# ---------------------------------------------------------------------------
def process_report(report_text: str):
    if not report_text.strip():
        yield (
            _pipeline_html(-1),
            '<div class="status-bar status-error"><div class="status-dot"></div>Please enter a radiology report.</div>',
            _concepts_html({}), _ontology_html({}), _categories_html({}), _studio_html("", {}), "",
        )
        return

    initial_state = {
        "input_report":                  report_text,
        "user_query":                    "structure the given medical report according to ABCDEF protocol",
        "input_findings":                "",
        "modules_queue":                 [],
        "concepts":                      {},
        "existing_categorized_concepts": {},
        "new_categorized_concepts":      {},
        "ontology_mapping":              {},
        "structured_report":             {},
    }

    yield (
        _pipeline_html(active_step=0),
        '<div class="status-bar status-running"><div class="status-dot"></div>Starting pipeline…</div>',
        _concepts_html({}), _ontology_html({}), _categories_html({}), _studio_html(report_text, {}), "",
    )

    pipeline_start = time.time()
    step_times: dict[str, float] = {}
    acc: dict = dict(initial_state)
    node_start = time.time()

    try:
        for event in full_agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                elapsed = time.time() - node_start
                node_start = time.time()

                if isinstance(node_output, dict):
                    acc.update(node_output)

                if node_name not in STEP_INDEX:
                    continue

                step_times[node_name] = elapsed
                next_active = STEP_INDEX[node_name] + 1

                concepts   = acc.get("concepts",                 {})
                ontology   = acc.get("ontology_mapping",         {})
                categories = acc.get("new_categorized_concepts", {})
                report     = acc.get("structured_report",        {})
                trace_map  = _trace_for(acc, report, args.trace_score_threshold)

                if next_active < len(PIPELINE_STEPS):
                    next_key   = PIPELINE_STEPS[next_active][0]
                    status_msg = STEP_LABEL.get(next_key, "Processing…")
                else:
                    status_msg = "Finalising…"

                yield (
                    _pipeline_html(active_step=next_active, step_times=step_times),
                    f'<div class="status-bar status-running"><div class="status-dot"></div>{status_msg}</div>',
                    _concepts_html(concepts), _ontology_html(ontology),
                    _categories_html(categories), _studio_html(report_text, report, trace_map), "",
                )

        total_time = time.time() - pipeline_start
        concepts   = acc.get("concepts",                 {})
        ontology   = acc.get("ontology_mapping",         {})
        categories = acc.get("new_categorized_concepts", {})
        report     = acc.get("structured_report",        {})
        trace_map  = _trace_for(acc, report, args.trace_score_threshold)
        debug      = json.dumps(
            {k: v for k, v in acc.items() if k != "modules_queue"},
            indent=2, default=str,
        )

        status = (
            f'<div class="status-bar status-done"><div class="status-dot"></div>'
            f'Pipeline complete &nbsp;·&nbsp; {len(concepts)} concepts &nbsp;·&nbsp; '
            f'total: <strong>{_fmt(total_time)}</strong></div>'
        )
        yield (
            _pipeline_html(done=True, step_times=step_times, total_time=total_time),
            status,
            _concepts_html(concepts), _ontology_html(ontology),
            _categories_html(categories), _studio_html(report_text, report, trace_map), debug,
        )

    except Exception as e:
        yield (
            _pipeline_html(-1),
            f'<div class="status-bar status-error"><div class="status-dot"></div>Error: {_esc(e)}</div>',
            _concepts_html({}), _ontology_html({}), _categories_html({}), _studio_html(report_text, {}),
            traceback.format_exc(),
        )


def clear_all():
    return (
        "",
        _pipeline_html(-1),
        '<div class="status-bar status-idle"><div class="status-dot"></div>Awaiting report input.</div>',
        _concepts_html({}), _ontology_html({}), _categories_html({}), _studio_html("", {}), "",
    )


SAMPLE = args.sample_report


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(css=CSS, title=args.title) as demo:

    gr.HTML(f"""
    <div class="app-header">
        <div class="header-logo">🫁</div>
        <div class="header-text">
            <h1>{_esc(args.title)}</h1>
            <p>MEDICAL PLAN · ACT · OBSERVE  ·  ABCDEF PROTOCOL STRUCTURING</p>
        </div>
        <div class="header-badge">Qwen3-4B · LoRA · SNOMED CT</div>
    </div>
    """)

    pipeline_html = gr.HTML(_pipeline_html(-1))
    status_html   = gr.HTML(
        '<div class="status-bar status-idle">'
        '<div class="status-dot"></div>Awaiting report input.</div>'
    )

    # ── INPUT ROW ──────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gr.HTML('<div class="section-label">Radiology Report Input</div>')
            report_input = gr.Textbox(
                placeholder="Paste a free-text radiology report here…",
                lines=8, max_lines=20, show_label=False,
            )
            with gr.Row():
                run_btn   = gr.Button("▶  Structure Report", variant="primary")
                clear_btn = gr.Button("✕  Clear",            variant="secondary")

        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">Example</div>')
            gr.HTML(f'<div class="panel-card">{_esc(SAMPLE)}</div>')
            load_btn = gr.Button("Load example report", variant="secondary")

    # ── LINKED STUDIO: source ↔ structured ─────────────────────────────────
    gr.HTML('<div class="full-width-divider"><div class="section-label">Transformation Studio</div></div>')
    gr.HTML(
        '<div class="hint-banner">'
        '<span class="hint-icon">🔗</span>'
        '<span>Hover a sentence on <b>either side</b>. Its counterpart lights up in the same colour '
        'and link number, everything else dims, and hovering a structured sentence opens its '
        'concept, SNOMED CT mapping, category and match confidence.</span>'
        '</div>'
    )
    studio_html = gr.HTML(_studio_html("", {}))

    # ── FULL-WIDTH TABS ────────────────────────────────────────────────────
    gr.HTML('<div class="full-width-divider"><div class="section-label">Analysis Details</div></div>')

    with gr.Tabs():
        with gr.Tab("Concepts"):
            gr.HTML('<div class="section-label" style="margin-top:12px;">Extracted Concepts → Source Sentences</div>')
            concepts_html = gr.HTML(_concepts_html({}))
        with gr.Tab("Ontology"):
            gr.HTML('<div class="section-label" style="margin-top:12px;">SNOMED CT Mappings</div>')
            ontology_html = gr.HTML(_ontology_html({}))
        with gr.Tab("Categories"):
            gr.HTML('<div class="section-label" style="margin-top:12px;">ABCDEF Classification</div>')
            categories_html = gr.HTML(_categories_html({}))
        with gr.Tab("Debug"):
            gr.HTML('<div class="section-label" style="margin-top:12px;">Full State JSON</div>')
            debug_box = gr.Code(language="json", label="", lines=20)

    gr.HTML(
        '<div class="app-footer">'
        '<span>MedPAO-Fast · Plan–Act–Observe agent for ABCDEF-protocol chest X-ray structuring</span>'
        '<span class="footer-links">'
        '<a href="https://github.com/MiRL-IITM/medpao-fast" target="_blank">GitHub</a>'
        ' &nbsp;·&nbsp; '
        '<a href="https://huggingface.co/shrishSVaidya/medpao-fast-models" target="_blank">LoRA adapters</a>'
        '</span>'
        '</div>'
    )

    outputs = [pipeline_html, status_html, concepts_html, ontology_html,
               categories_html, studio_html, debug_box]

    run_btn.click(fn=process_report, inputs=[report_input], outputs=outputs)
    load_btn.click(fn=lambda: SAMPLE, outputs=[report_input])
    clear_btn.click(fn=clear_all, outputs=[report_input] + outputs)


if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)