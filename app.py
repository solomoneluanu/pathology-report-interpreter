"""
PathPal - Pathology Report Interpreter for Patients
====================================================
An educational tool that helps patients understand pathology reports by:
  1. Highlighting medical terms found in the report (biomedical NER)
  2. Narrating a plain-language explanation, built from detected entities and
     a curated glossary (not a generative model, so nothing is fabricated)
  3. Explaining common pathology jargon with a built-in glossary
  4. (Multimodal) Accepting a photo/scan of a printed report via OCR

Models used (Hugging Face Hub):
  - d4data/biomedical-ner-all      -> token classification (NER)

DISCLAIMER: This app is for educational purposes only. It is NOT a medical
device and does NOT provide medical advice. Patients should always discuss
their results with their healthcare provider.

Author: <YOUR NAME HERE>
Course project - Transformer-Based Application using Gradio & HF Spaces
"""

import json
import re
from pathlib import Path

import gradio as gr
from transformers import pipeline

# ---------------------------------------------------------------------------
# 1. Load models once at startup (small models chosen so the free CPU tier
#    on Hugging Face Spaces can run them without errors).
# ---------------------------------------------------------------------------
print("Loading biomedical NER model...")
ner_pipeline = pipeline(
    "token-classification",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",  # merge sub-word tokens into full entities
)

# ---------------------------------------------------------------------------
# 2. Load the plain-language glossary (curated JSON bundled with the app).
#    This part is deterministic: it never depends on model behavior.
# ---------------------------------------------------------------------------
GLOSSARY_PATH = Path(__file__).parent / "glossary.json"
with open(GLOSSARY_PATH, encoding="utf-8") as f:
    # Most entries are a plain definition string. A term may instead be
    # {"definition": ..., "short": ...} when it has a curated short gloss.
    GLOSSARY: dict[str, str | dict] = json.load(f)

# Pre-compile a regex per glossary term for whole-word, case-insensitive match.
GLOSSARY_PATTERNS = {
    term: re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for term in GLOSSARY
}


def glossary_definition(term: str) -> str:
    """Full definition text for a glossary term, regardless of entry shape."""
    entry = GLOSSARY[term]
    return entry["definition"] if isinstance(entry, dict) else entry


def glossary_short_gloss(term: str) -> str:
    """Short parenthetical gloss for a term: the curated 'short' field if
    present, otherwise the first sentence of the full definition."""
    entry = GLOSSARY[term]
    if isinstance(entry, dict) and entry.get("short"):
        return entry["short"].rstrip(".!? ")
    return re.split(r"(?<=[.!?])\s+", glossary_definition(term).strip())[0].rstrip(".!? ")

# Friendly display names for the NER entity groups we care about most.
ENTITY_LABELS = {
    "Disease_disorder": "Disease / disorder",
    "Sign_symptom": "Sign / symptom",
    "Biological_structure": "Body part / tissue",
    "Diagnostic_procedure": "Diagnostic procedure",
    "Therapeutic_procedure": "Treatment / procedure",
    "Medication": "Medication",
    "Lab_value": "Lab value",
    "Detailed_description": "Descriptive finding",
    "Severity": "Severity",
}

DISCLAIMER_MD = (
    "> ⚠️ **Important:** This tool is for **education only**. It is not a "
    "medical device, it can make mistakes, and it does not replace your "
    "doctor. Always discuss your pathology results with your healthcare "
    "provider."
)


# ---------------------------------------------------------------------------
# 3. Core analysis functions
# ---------------------------------------------------------------------------
def highlight_entities(report_text: str, entities: list):
    """Build spans for gr.HighlightedText from precomputed NER entities."""
    spans = []
    cursor = 0
    # Sort by start position and rebuild the text with labeled spans.
    for ent in sorted(entities, key=lambda e: e["start"]):
        start, end = ent["start"], ent["end"]
        if start < cursor:  # skip overlapping entities
            continue
        label = ENTITY_LABELS.get(ent["entity_group"], ent["entity_group"])
        if start > cursor:
            spans.append((report_text[cursor:start], None))
        spans.append((report_text[start:end], label))
        cursor = end
    if cursor < len(report_text):
        spans.append((report_text[cursor:], None))
    return spans


def build_glossary_table(report_text: str) -> str:
    """Find glossary terms present in the report and explain them in lay terms."""
    rows = []
    for term, pattern in GLOSSARY_PATTERNS.items():
        if pattern.search(report_text):
            rows.append(f"| **{term.title()}** | {glossary_definition(term)} |")

    if not rows:
        return (
            "No common pathology terms from the built-in glossary were found "
            "in this report."
        )
    header = "| Term in your report | What it means in plain English |\n|---|---|\n"
    return header + "\n".join(rows)


# Multi-word entries first (by word count, then char length) so overlapping
# phrases like "invasive ductal carcinoma" resolve to one combined term
# instead of matching each component word ("invasive", "ductal", "carcinoma").
GLOSSARY_TERMS_BY_SPECIFICITY = sorted(
    GLOSSARY, key=lambda term: (-len(term.split()), -len(term))
)

DIFFERENTIATION_TERMS = [
    "poorly differentiated",
    "moderately differentiated",
    "well differentiated",
    "differentiated",
]
RECEPTOR_TERMS = ["estrogen receptor", "progesterone receptor", "her2"]
RECEPTOR_LABELS = {
    "estrogen receptor": "estrogen receptor",
    "progesterone receptor": "progesterone receptor",
    "her2": "HER2",
}


def _first_entity_text(entities: list, report_text: str, group: str) -> str | None:
    """Text of the earliest-occurring entity in the given NER group, or None."""
    matches = [e for e in entities if e["entity_group"] == group]
    if not matches:
        return None
    first = min(matches, key=lambda e: e["start"])
    return report_text[first["start"] : first["end"]].strip()


def _find_glossary_term(text: str) -> str | None:
    """Most specific glossary term (multi-word first) found in text, or None."""
    for term in GLOSSARY_TERMS_BY_SPECIFICITY:
        if GLOSSARY_PATTERNS[term].search(text):
            return term
    return None


def _lowercase_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


DIAGNOSIS_SECTION_LABELS = ["Diagnosis", "Final diagnosis", "Impression"]
SPECIMEN_SECTION_LABELS = ["Specimen"]


def _extract_section_text(report_text: str, labels: list) -> str | None:
    """Deterministically extract a labeled section's content (e.g.
    "Diagnosis:" or "Specimen:"), independent of the NER model. Tries labels
    in priority order and stops at the next blank line, the next apparent
    section header, or the end of the report."""
    for label in labels:
        pattern = re.compile(
            rf"^[ \t]*{re.escape(label)}[ \t]*:[ \t]*(.+?)"
            r"(?=\n[ \t]*\n|\n[ \t]*[A-Z][A-Za-z /]{2,40}:|\Z)",
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(report_text)
        if match:
            text = re.sub(r"\s+", " ", match.group(1)).strip()
            if text:
                return text
    return None


def _extract_diagnosis_text(report_text: str) -> str | None:
    """The report's Diagnosis / Final diagnosis / Impression text, so the
    main finding is never missed just because NER failed to tag it."""
    return _extract_section_text(report_text, DIAGNOSIS_SECTION_LABELS)


def _extract_specimen_text(report_text: str) -> str | None:
    """The report's Specimen text, so the sample description is never
    guessed together from mistagged NER entities."""
    return _extract_section_text(report_text, SPECIMEN_SECTION_LABELS)


def narrate_report(report_text: str, entities: list) -> str:
    """Build a flowing plain-language paragraph from detected NER entities and
    glossary terms. Every sentence is grounded in something actually found in
    the report; nothing is stated unless backed by an entity or a matched
    glossary term."""
    sentences = []

    # 1. Opening: the report's own Diagnosis / Final diagnosis / Impression
    # text, parsed deterministically (no NER dependency), so the main
    # finding always appears even if the NER model fails to tag it. This is
    # authoritative report text, so it's safe to state directly; only the
    # glossary meaning woven in afterward needs a matched term to back it.
    diagnosis_text = _extract_diagnosis_text(report_text)
    if diagnosis_text:
        sentence = f"According to your report, the diagnosis is: {diagnosis_text}"
        term = _find_glossary_term(diagnosis_text)
        if term:
            sentence += f" In plain terms, {_lowercase_first(glossary_short_gloss(term))}."
        sentences.append(sentence)

    # 2. Specimen, parsed deterministically from the report's "Specimen:"
    # line (same approach as the diagnosis, item 1) rather than assembled
    # from NER entities, which could mistag the report's title line and
    # produce broken fragments like "a surgical of the breast". Omitted
    # entirely if the report has no "Specimen:" line, rather than guessing.
    specimen_text = _extract_specimen_text(report_text)
    if specimen_text:
        sentences.append(f"The sample examined was: {specimen_text}")

    # 3. Main diagnosis via NER -- only as a fallback for reports with no
    # deterministic Diagnosis/Final diagnosis/Impression section (item 1).
    if not diagnosis_text:
        diagnosis_entity = _first_entity_text(entities, report_text, "Disease_disorder")
        if diagnosis_entity:
            term = _find_glossary_term(diagnosis_entity) or _find_glossary_term(report_text)
            if term:
                sentences.append(
                    f"The main finding is {diagnosis_entity.lower()}, which means "
                    f"{_lowercase_first(glossary_short_gloss(term))}."
                )
            else:
                sentences.append(f"The main finding is {diagnosis_entity.lower()}.")

    # 4. Grade / differentiation, in plain terms.
    diff_term = next(
        (t for t in DIFFERENTIATION_TERMS if GLOSSARY_PATTERNS[t].search(report_text)), None
    )
    if diff_term:
        sentences.append(
            f"The cells are described as {diff_term}, which means "
            f"{_lowercase_first(glossary_short_gloss(diff_term))}."
        )

    # 5. Receptor status, in plain terms. Each clause names its receptor so
    # near-identical glossary wording (e.g. ER vs. PR) doesn't read as a
    # repeated, unattributed fragment.
    receptor_clauses = [
        f"{RECEPTOR_LABELS[t]} testing looks for {_lowercase_first(glossary_short_gloss(t))}"
        for t in RECEPTOR_TERMS
        if GLOSSARY_PATTERNS[t].search(report_text)
    ]
    if receptor_clauses:
        if len(receptor_clauses) == 1:
            joined = receptor_clauses[0]
        else:
            joined = ", ".join(receptor_clauses[:-1]) + ", and " + receptor_clauses[-1]
        sentences.append(f"The report also checked receptor status: {joined}.")

    # 6. Closing reminder -- general advice, not a claim about the report.
    sentences.append(
        "As always, share this report with your doctor, who can explain what "
        "it means for your specific care."
    )

    return " ".join(sentences)


def analyze_report(report_text: str):
    """Full pipeline: NER highlights + summary + glossary."""
    report_text = (report_text or "").strip()
    if not report_text:
        raise gr.Error("Please paste a pathology report (or load an example) first.")
    if len(report_text) < 40:
        raise gr.Error(
            "That text looks too short to be a pathology report. "
            "Please paste the full report text."
        )

    entities = ner_pipeline(report_text)
    highlighted = highlight_entities(report_text, entities)
    narration = narrate_report(report_text, entities)
    glossary_md = build_glossary_table(report_text)
    return highlighted, narration, glossary_md


def ocr_report_photo(image) -> str:
    """(Multimodal) Extract text from a photo/scan of a printed report."""
    if image is None:
        raise gr.Error("Please upload a photo or scan of your report first.")
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise gr.Error("OCR dependency is missing on this deployment.") from exc

    text = pytesseract.image_to_string(image)
    text = text.strip()
    if not text:
        raise gr.Error(
            "No text could be read from that image. Try a sharper, well-lit "
            "photo taken straight-on."
        )
    return text


# ---------------------------------------------------------------------------
# 4. Example inputs (synthetic reports - no real patient data)
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path(__file__).parent / "sample_inputs"
EXAMPLES = []
if SAMPLES_DIR.exists():
    for sample_file in sorted(SAMPLES_DIR.glob("*.txt")):
        EXAMPLES.append([sample_file.read_text(encoding="utf-8").strip()])


# ---------------------------------------------------------------------------
# 5. Gradio interface -- always-on dark, modern/clinical theme
# ---------------------------------------------------------------------------
THEME = (
    gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    .set(
        # Surfaces. Light and dark variants are set to the SAME dark palette
        # (rather than relying only on the forced ".dark" class below) so the
        # UI can never flash or fall back to a light theme.
        body_background_fill="#0F172A",
        body_background_fill_dark="#0F172A",
        background_fill_primary="#0F172A",
        background_fill_primary_dark="#0F172A",
        background_fill_secondary="#1E293B",
        background_fill_secondary_dark="#1E293B",
        block_background_fill="#1E293B",
        block_background_fill_dark="#1E293B",
        panel_background_fill="#1E293B",
        panel_background_fill_dark="#1E293B",
        input_background_fill="#0F172A",
        input_background_fill_dark="#0F172A",
        # Borders
        block_border_color="#334155",
        block_border_color_dark="#334155",
        border_color_primary="#334155",
        border_color_primary_dark="#334155",
        panel_border_color="#334155",
        panel_border_color_dark="#334155",
        input_border_color="#334155",
        input_border_color_dark="#334155",
        input_border_color_focus="#2DD4BF",
        input_border_color_focus_dark="#2DD4BF",
        # Text
        body_text_color="#F1F5F9",
        body_text_color_dark="#F1F5F9",
        body_text_color_subdued="#94A3B8",
        body_text_color_subdued_dark="#94A3B8",
        body_text_size="16px",
        block_label_text_color="#94A3B8",
        block_label_text_color_dark="#94A3B8",
        block_title_text_color="#F1F5F9",
        block_title_text_color_dark="#F1F5F9",
        # Accent / links
        color_accent="#2DD4BF",
        color_accent_soft="rgba(45, 212, 191, 0.15)",
        color_accent_soft_dark="rgba(45, 212, 191, 0.15)",
        border_color_accent="#2DD4BF",
        border_color_accent_dark="#2DD4BF",
        link_text_color="#2DD4BF",
        link_text_color_dark="#2DD4BF",
        link_text_color_hover="#5EEAD4",
        link_text_color_hover_dark="#5EEAD4",
        # Buttons -- primary (teal accent)
        button_primary_background_fill="#2DD4BF",
        button_primary_background_fill_dark="#2DD4BF",
        button_primary_background_fill_hover="#5EEAD4",
        button_primary_background_fill_hover_dark="#5EEAD4",
        button_primary_border_color="#2DD4BF",
        button_primary_border_color_dark="#2DD4BF",
        button_primary_text_color="#0F172A",
        button_primary_text_color_dark="#0F172A",
        # Buttons -- secondary (visually lighter/quieter)
        button_secondary_background_fill="#334155",
        button_secondary_background_fill_dark="#334155",
        button_secondary_background_fill_hover="#475569",
        button_secondary_background_fill_hover_dark="#475569",
        button_secondary_border_color="#334155",
        button_secondary_border_color_dark="#334155",
        button_secondary_text_color="#F1F5F9",
        button_secondary_text_color_dark="#F1F5F9",
        # Radii
        block_radius="12px",
        container_radius="12px",
        input_radius="8px",
        button_large_radius="10px",
        button_medium_radius="10px",
        button_small_radius="8px",
    )
)

# Forces dark mode regardless of the visitor's OS/browser preference. Belt
# and suspenders with the identical light/dark palette above: Gradio's own
# theme CSS is gated behind ":root.dark, :root .dark", and some component
# internals key off that class too, so it's still added explicitly.
FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"

CUSTOM_CSS = """
:root, .dark {
  --pp-bg: #0F172A;
  --pp-surface: #1E293B;
  --pp-border: #334155;
  --pp-text: #F1F5F9;
  --pp-muted: #94A3B8;
  --pp-accent: #2DD4BF;
  --pp-amber: #F59E0B;
}

html, body, .gradio-container {
  background: var(--pp-bg) !important;
  color: var(--pp-text);
  font-size: 16px;
}

/* Let the page scroll naturally instead of trapping content in a small
   inner scroll region. Gradio's app shell defaults to a fixed-height flex
   column with overflow:hidden, which pushes any overflow into a nested
   scrollbar rather than growing the page. */
html, body {
  height: auto !important;
  min-height: 100%;
}
.gradio-container,
[class*="gradio-container"] {
  overflow: visible !important;
  height: auto !important;
  min-height: 100vh;
}

/* ---- Header band ---- */
.pp-header { padding: 8px 4px 4px; }
#app-title h1 {
  font-size: 2.25rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  margin-bottom: 0.15rem;
  color: var(--pp-text);
}
#app-tagline p {
  color: var(--pp-muted);
  font-size: 1.05rem;
  line-height: 1.5;
  margin-top: 0;
}
#disclaimer blockquote {
  border-left: 4px solid var(--pp-amber);
  background: rgba(245, 158, 11, 0.08);
  border-radius: 8px;
  padding: 14px 18px;
  margin: 16px 0 0;
}
#disclaimer blockquote p {
  margin: 0;
  color: var(--pp-text);
  font-size: 1rem;
}

/* ---- Cards ---- */
.pp-card {
  background: var(--pp-surface) !important;
  border: 1px solid var(--pp-border) !important;
  border-radius: 12px !important;
  padding: 24px !important;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
  margin-bottom: 20px;
}
#results-heading h2 { margin-top: 0; color: var(--pp-text); }

/* ---- Buttons ---- */
#analyze-btn button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 14px rgba(45, 212, 191, 0.25);
  transition: filter 0.15s ease, transform 0.1s ease;
}
#analyze-btn button:hover { filter: brightness(1.08); transform: translateY(-1px); }
#analyze-btn button:active { transform: translateY(0); }

#ocr-btn button {
  background: transparent !important;
  border: 1px solid var(--pp-border) !important;
  color: var(--pp-muted) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  transition: background-color 0.15s ease, color 0.15s ease;
}
#ocr-btn button:hover {
  background: rgba(148, 163, 184, 0.1) !important;
  color: var(--pp-text) !important;
}

/* ---- HighlightedText legend as rounded pill/chips. Gradio doesn't expose
   stable public class names for this internal markup, so these are
   deliberately broad, best-effort selectors. ---- */
#hl-output .categories,
#hl-output [class*="legend" i],
#hl-output [class*="categor" i] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
}
#hl-output .categories > *,
#hl-output [class*="legend" i] > *,
#hl-output [class*="categor" i] > * {
  border-radius: 9999px !important;
  padding: 4px 14px !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  border: 1px solid var(--pp-border) !important;
}

/* ---- Accessibility ---- */
*:focus-visible {
  outline: 2px solid var(--pp-accent) !important;
  outline-offset: 2px !important;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}
"""

with gr.Blocks(
    title="PathPal - Pathology Report Interpreter",
    theme=THEME,
    css=CUSTOM_CSS,
    js=FORCE_DARK_JS,
) as demo:
    with gr.Column(elem_classes=["pp-header"]):
        gr.Markdown("# 🔬 PathPal — Pathology Report Interpreter", elem_id="app-title")
        gr.Markdown(
            "Paste your pathology report (or upload a photo of it) and PathPal will "
            "**highlight the medical terms**, give you a **plain-language explanation**, "
            "and **explain common jargon** — so you can walk into your next "
            "appointment with better questions.",
            elem_id="app-tagline",
        )
        gr.Markdown(DISCLAIMER_MD, elem_id="disclaimer")

    with gr.Column(elem_classes=["pp-card"]):
        with gr.Tabs():
            # --------------------- TAB 1: paste text ------------------------
            with gr.Tab("📋 Paste report text"):
                report_input = gr.Textbox(
                    label="Pathology report text",
                    placeholder="Paste the full text of the pathology report here...",
                    lines=18,
                )
                analyze_btn = gr.Button(
                    "Explain my report", variant="primary", elem_id="analyze-btn"
                )

            # ----------------- TAB 2: photo upload (OCR) --------------------
            with gr.Tab("📷 Upload a photo of the report (multimodal)"):
                gr.Markdown(
                    "Take a clear, well-lit photo of the printed report, straight-on. "
                    "The extracted text will appear in the box on the left tab — "
                    "review it for OCR errors, then click **Explain my report**."
                )
                image_input = gr.Image(
                    label="Photo or scan of the printed report", type="pil"
                )
                ocr_btn = gr.Button(
                    "Read text from photo", variant="secondary", elem_id="ocr-btn"
                )

    with gr.Column(elem_classes=["pp-card"]):
        gr.Markdown("## Results", elem_id="results-heading")
        with gr.Row():
            with gr.Column(scale=3):
                highlighted_output = gr.HighlightedText(
                    label="Your report with medical terms highlighted",
                    combine_adjacent=True,
                    show_legend=True,
                    elem_id="hl-output",
                )
            with gr.Column(scale=2):
                summary_output = gr.Textbox(
                    label="What this report means (plain-language explanation)",
                    lines=6,
                )
        glossary_output = gr.Markdown(label="Glossary of terms found in your report")

    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            inputs=[report_input],
            label="Example reports (synthetic — not real patients)",
        )

    # ------------------------------ events -----------------------------------
    analyze_btn.click(
        fn=analyze_report,
        inputs=[report_input],
        outputs=[highlighted_output, summary_output, glossary_output],
    )
    ocr_btn.click(
        fn=ocr_report_photo,
        inputs=[image_input],
        outputs=[report_input],
    )

    gr.Markdown(
        "---\n*Models: [d4data/biomedical-ner-all]"
        "(https://huggingface.co/d4data/biomedical-ner-all) · "
        "plain-language narration is rule-based (entities + curated glossary, "
        "no generative model) · OCR via Tesseract. Built with 🤗 Transformers "
        "and Gradio.*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, show_error=True)
