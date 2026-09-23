"""
MedPAO-Fast — Gradio Interface
Run: python gradio_app.py

Requires langgraph_agent.py (with full_agent compiled graph) in the same directory.
"""
import os
import time

os.environ['HF_HOME'] = '/media/shrish/Data/huggingface_models' # set the huggingface cache directory to a custom path
import gradio as gr
import json
import time
import traceback

# ---------------------------------------------------------------------------
# Import the compiled LangGraph agent.
# Model loading + LoRA init happens here at startup (once).
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

# Map node name → position index in PIPELINE_STEPS
STEP_INDEX = {key: i for i, (key, _, _) in enumerate(PIPELINE_STEPS)}

# Human-readable "now running" labels shown in the status bar
STEP_LABEL = {
    "get_concept":                "Extracting concepts…",
    "check_cache":                "Checking cache…",
    "ontology_mapping":           "Mapping ontologies…",
    "categorize_concepts":        "Classifying ABCDEF…",
    "generate_structured_report": "Generating report…",
}


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def _fmt(s: float) -> str:
    return f"{s:.2f}s"


def _pipeline_html(
    active_step: int = -1,
    done: bool = False,
    step_times: dict = None,
    total_time: float = None,
) -> str:
    step_times = step_times or {}
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
        connector = '<div class="connector"></div>' if i < len(PIPELINE_STEPS) - 1 else ""
        items += f"""
        <div class="pipeline-item">
            <div class="step-bubble {state_cls}">{icon}</div>
            <div class="step-label {state_cls}">{label}</div>
            {time_badge}
        </div>{connector}"""

    total_row = ""
    if total_time is not None:
        total_row = f"""
        <div class="pipeline-total">
            <span class="total-label">⏱ Total pipeline time</span>
            <span class="total-value">{_fmt(total_time)}</span>
        </div>"""

    return f'<div class="pipeline-wrap"><div class="pipeline-track">{items}</div>{total_row}</div>'


def _concepts_html(concepts: dict) -> str:
    if not concepts:
        return '<p class="empty-msg">No concepts extracted yet.</p>'
    rows = "".join(f"""
        <div class="concept-row">
            <span class="concept-tag">{c}</span>
            <span class="concept-src">↳ {s}</span>
        </div>""" for c, s in concepts.items())
    return f'<div class="concepts-grid">{rows}</div>'


def _report_html(report: dict) -> str:
    if not report:
        return '<p class="empty-msg">Report not generated yet.</p>'
    cards = ""
    for key, text in report.items():
        label, icon, color = SECTION_META.get(key, (key, "📋", "#94A3B8"))
        no_finding = text.strip().lower() in ("no findings.", "no findings", "no findings identified.")
        dim = " dim" if no_finding else ""
        cards += f"""
        <div class="report-card{dim}">
            <div class="card-header" style="border-left:3px solid {color};">
                <span class="card-icon">{icon}</span>
                <span class="card-key" style="color:{color};">{key}</span>
                <span class="card-label">{label}</span>
            </div>
            <p class="card-text">{text}</p>
        </div>"""
    return f'<div class="report-grid">{cards}</div>'


def _ontology_html(onto: dict) -> str:
    if not onto:
        return '<p class="empty-msg">No ontology data.</p>'
    rows = "".join(f"""
        <div class="onto-row">
            <div class="onto-concept">{c}</div>
            <div class="onto-info">{info}</div>
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
            f'<span class="cat-chip" style="border-color:{color};color:{color};">{c}</span>'
            for c in buckets[cat]
        )
        rows += f"""
        <div class="cat-row">
            <div class="cat-badge" style="background:{color}22;color:{color};border:1px solid {color}55;">
                {icon} {cat} · {label}
            </div>
            <div class="cat-chips">{chips}</div>
        </div>"""
    return f'<div class="cat-list">{rows}</div>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0B0F1A;
    --surface:   #111827;
    --surface2:  #1A2235;
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
}

* { box-sizing:border-box; margin:0; padding:0; }

body, .gradio-container, .gradio-container *,
.svelte-1ipelgc, .prose, fieldset, legend,
.wrap, .label-wrap, .output-class {
    background-color: transparent;
    color: var(--text) !important;
    font-family: var(--fb) !important;
}
body, .gradio-container { background: var(--bg) !important; }

.app-header {
    display:flex; align-items:center; gap:18px;
    margin-bottom:36px; padding-bottom:24px;
    border-bottom:1px solid var(--border);
}
.header-logo {
    width:52px; height:52px;
    background:linear-gradient(135deg,#0EA5E9,#6366F1);
    border-radius:14px; display:flex; align-items:center; justify-content:center;
    font-size:26px; box-shadow:0 0 28px #0EA5E940; flex-shrink:0;
}
.header-text h1 {
    font-family:var(--fh) !important; font-size:2rem;
    color:#F8FAFC !important; letter-spacing:-0.5px; line-height:1;
}
.header-text p {
    font-size:0.74rem; color:var(--text-mid) !important;
    margin-top:5px; font-family:var(--fm) !important; letter-spacing:0.06em;
}
.header-badge {
    margin-left:auto; padding:6px 14px;
    border:1px solid #0EA5E940; border-radius:20px;
    font-size:0.72rem; font-family:var(--fm) !important;
    color:var(--accent) !important; background:#0EA5E910;
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
    box-shadow:0 4px 20px #6366F130 !important;
    transition:opacity .2s,transform .1s !important;
}
button[variant="primary"]:hover { opacity:.88 !important; transform:translateY(-1px) !important; }
button[variant="secondary"], .gr-button-secondary {
    background:var(--surface2) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    font-family:var(--fb) !important;
    border-radius:var(--r) !important;
    font-size:0.85rem !important; padding:10px 20px !important;
}
button[variant="secondary"]:hover { border-color:var(--accent) !important; color:var(--accent) !important; }

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

.pipeline-wrap {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--r); margin-bottom:16px; overflow:hidden;
}
.pipeline-track {
    display:flex; align-items:flex-start;
    padding:22px 18px 14px; overflow-x:auto; gap:0;
}
.pipeline-item { display:flex; flex-direction:column; align-items:center; gap:6px; min-width:92px; }
.connector { flex:1; height:1px; background:var(--border); min-width:10px; margin-bottom:44px; margin-top:18px; }

.step-bubble {
    width:36px; height:36px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:var(--fm); font-size:.72rem; font-weight:500;
    transition:all .3s; flex-shrink:0;
}
.step-pending { background:var(--surface2); color:var(--text-mid) !important; border:1px solid var(--border); }
.step-active  { background:#0EA5E920; color:var(--accent) !important; border:1px solid var(--accent); box-shadow:0 0 14px #38BDF840; animation:pulse 1.4s infinite; }
.step-done    { background:#10B98120; color:var(--success) !important; border:1px solid var(--success); }
@keyframes pulse { 0%,100%{box-shadow:0 0 8px #38BDF840} 50%{box-shadow:0 0 22px #38BDF880} }

.step-label { font-size:.64rem; font-family:var(--fm); text-align:center; line-height:1.3; }
.step-label.step-pending { color:var(--text-mid) !important; }
.step-label.step-active  { color:var(--accent)    !important; }
.step-label.step-done    { color:var(--success)   !important; }

.step-time {
    font-family:var(--fm); font-size:.62rem;
    color:var(--accent2) !important; text-align:center; min-height:14px;
}

.pipeline-total {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 20px; border-top:1px solid var(--border); background:#0EA5E908;
}
.total-label { font-family:var(--fm); font-size:.72rem; color:var(--text-mid) !important; letter-spacing:.06em; }
.total-value { font-family:var(--fm); font-size:.92rem; font-weight:600; color:var(--accent2) !important; letter-spacing:.04em; }

.status-bar {
    padding:10px 16px; border-radius:8px;
    font-family:var(--fm); font-size:.78rem;
    margin-bottom:16px; display:flex; align-items:center; gap:8px;
}
.status-idle    { background:#1E293B; color:var(--text-mid) !important; border:1px solid var(--border); }
.status-running { background:#0EA5E915; color:var(--accent) !important; border:1px solid #0EA5E930; }
.status-done    { background:#10B98115; color:var(--success) !important; border:1px solid #10B98130; }
.status-error   { background:#F8717115; color:var(--danger)  !important; border:1px solid #F8717130; }
.status-bar strong { color:var(--accent2) !important; }
.status-dot { width:7px; height:7px; border-radius:50%; background:currentColor; flex-shrink:0; }

.concepts-grid { display:flex; flex-direction:column; gap:8px; }
.concept-row {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; padding:10px 14px;
    display:flex; flex-direction:column; gap:5px;
}
.concept-tag { font-family:var(--fm); font-size:.8rem; color:var(--accent) !important; font-weight:500; }
.concept-src { font-size:.77rem; color:var(--text) !important; line-height:1.5; }

.report-grid {
    display:grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap:12px;
}
.report-card {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:10px; padding:16px 18px; transition:border-color .2s;
}
.report-card:hover { border-color:#2A4A7F; }
.report-card.dim   { opacity:.35; }
.card-header { display:flex; align-items:center; gap:10px; padding-left:12px; margin-bottom:10px; }
.card-icon   { font-size:1rem; }
.card-key    { font-family:var(--fm); font-weight:600; font-size:.95rem; }
.card-label  { font-size:.69rem; color:var(--text-mid) !important; font-family:var(--fm); text-transform:uppercase; letter-spacing:.08em; margin-left:4px; }
.card-text   { font-size:.85rem; color:var(--text) !important; line-height:1.7; padding-left:12px; }

.onto-list { display:flex; flex-direction:column; gap:10px; }
.onto-row  { background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; }
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
    padding:18px; text-align:center;
    border:1px dashed var(--border); border-radius:8px;
}

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
"""


# ---------------------------------------------------------------------------
# Processing function
# Streams LangGraph node completions → updates pipeline + tabs in real-time.
# ---------------------------------------------------------------------------
def process_report(report_text: str):
    if not report_text.strip():
        yield (
            _pipeline_html(-1),
            '<div class="status-bar status-error"><div class="status-dot"></div>Please enter a radiology report.</div>',
            _concepts_html({}), _ontology_html({}), _categories_html({}), _report_html({}), "",
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

    # Show step 0 as active immediately, before the first blocking LLM call
    yield (
        _pipeline_html(active_step=0),
        '<div class="status-bar status-running"><div class="status-dot"></div>Starting pipeline…</div>',
        _concepts_html({}), _ontology_html({}), _categories_html({}), _report_html({}), "",
    )

    pipeline_start = time.time()
    step_times: dict[str, float] = {}
    # Accumulates the full agent state as node outputs stream in
    acc: dict = dict(initial_state)
    node_start = time.time()

    try:
        # stream_mode="updates" yields {node_name: output_dict} after each node
        for event in full_agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                elapsed = time.time() - node_start
                node_start = time.time()

                # Merge this node's output into our accumulated state view
                if isinstance(node_output, dict):
                    acc.update(node_output)

                # planner / orchestrator are not in the pipeline UI — skip
                if node_name not in STEP_INDEX:
                    continue

                step_times[node_name] = elapsed

                # Point the active highlight at the NEXT step to run
                next_active = STEP_INDEX[node_name] + 1

                # Pull latest data from accumulated state for live tab updates
                concepts   = acc.get("concepts",                 {})
                ontology   = acc.get("ontology_mapping",         {})
                categories = acc.get("new_categorized_concepts", {})
                report     = acc.get("structured_report",        {})

                # Status bar label: show the step that is about to run
                if next_active < len(PIPELINE_STEPS):
                    next_key   = PIPELINE_STEPS[next_active][0]
                    status_msg = STEP_LABEL.get(next_key, "Processing…")
                else:
                    status_msg = "Finalising…"

                yield (
                    _pipeline_html(active_step=next_active, step_times=step_times),
                    f'<div class="status-bar status-running"><div class="status-dot"></div>{status_msg}</div>',
                    _concepts_html(concepts), _ontology_html(ontology),
                    _categories_html(categories), _report_html(report), "",
                )

        # ── All nodes done — render final state ───────────────────────────
        total_time = time.time() - pipeline_start
        concepts   = acc.get("concepts",                 {})
        ontology   = acc.get("ontology_mapping",         {})
        categories = acc.get("new_categorized_concepts", {})
        report     = acc.get("structured_report",        {})
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
            _categories_html(categories), _report_html(report), debug,
        )

    except Exception as e:
        yield (
            _pipeline_html(-1),
            f'<div class="status-bar status-error"><div class="status-dot"></div>Error: {e}</div>',
            _concepts_html({}), _ontology_html({}), _categories_html({}), _report_html({}),
            traceback.format_exc(),
        )


def clear_all():
    return (
        "",
        _pipeline_html(-1),
        '<div class="status-bar status-idle"><div class="status-dot"></div>Awaiting report input.</div>',
        _concepts_html({}), _ontology_html({}), _categories_html({}), _report_html({}), "",
    )


SAMPLE = (
    "PA and lateral views of the chest are submitted. Lungs appear well inflated "
    "without evidence of focal airspace consolidation, pleural effusions, pulmonary "
    "edema, or pneumothorax. Cardiac and mediastinal contours are within normal limits. "
    "No acute bony abnormality is appreciated."
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(css=CSS, title="MedPAO-Fast") as demo:

    gr.HTML("""
    <div class="app-header">
        <div class="header-logo">🫁</div>
        <div class="header-text">
            <h1>MedPAO-Fast</h1>
            <p>MEDICAL PLAN · ACT · OBSERVE  ·  ABCDEF PROTOCOL STRUCTURING</p>
        </div>
        <div class="header-badge">Qwen3-4B · LoRA · SNOMED CT</div>
    </div>
    """)

    # ── TOP ROW: Input (left) + Structured Report (right) ─────────────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">Radiology Report Input</div>')
            report_input = gr.Textbox(
                placeholder="Paste a free-text radiology report here…",
                lines=10, max_lines=20, show_label=False,
            )
            with gr.Row():
                run_btn   = gr.Button("▶  Structure Report", variant="primary")
                clear_btn = gr.Button("✕  Clear",            variant="secondary")

            gr.HTML('<div class="section-label" style="margin-top:16px;">Example</div>')
            gr.HTML(f'<div class="panel-card">{SAMPLE}</div>')
            load_btn = gr.Button("Load example report", variant="secondary")

            gr.HTML('<div class="section-label" style="margin-top:22px;">Pipeline</div>')
            pipeline_html = gr.HTML(_pipeline_html(-1))
            status_html   = gr.HTML(
                '<div class="status-bar status-idle">'
                '<div class="status-dot"></div>Awaiting report input.</div>'
            )

        with gr.Column(scale=3):
            gr.HTML('<div class="section-label">Structured Report</div>')
            report_html = gr.HTML(_report_html({}))

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

    outputs = [pipeline_html, status_html, concepts_html, ontology_html,
               categories_html, report_html, debug_box]

    run_btn.click(fn=process_report, inputs=[report_input], outputs=outputs)
    load_btn.click(fn=lambda: SAMPLE, outputs=[report_input])
    clear_btn.click(fn=clear_all, outputs=[report_input] + outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)