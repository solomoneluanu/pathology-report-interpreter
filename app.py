"""
PathPal - Pathology Report Interpreter for Patients
====================================================
An educational tool that helps patients understand pathology reports by:
  1. Highlighting medical terms found in the report (biomedical NER)
  2. Generating a plain-language summary (medical summarization model)
  3. Explaining common pathology jargon with a built-in glossary
  4. (Multimodal) Accepting a photo/scan of a printed report via OCR

Models used (Hugging Face Hub):
  - d4data/biomedical-ner-all      -> token classification (NER)
  - sshleifer/distilbart-cnn-12-6  -> BART-based summarization

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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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

print("Loading medical summarization model...")
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)

# ---------------------------------------------------------------------------
# 2. Load the plain-language glossary (curated JSON bundled with the app).
#    This part is deterministic: it never depends on model behavior.
# ---------------------------------------------------------------------------
GLOSSARY_PATH = Path(__file__).parent / "glossary.json"
with open(GLOSSARY_PATH, encoding="utf-8") as f:
    GLOSSARY: dict[str, str] = json.load(f)

# Pre-compile a regex per glossary term for whole-word, case-insensitive match.
GLOSSARY_PATTERNS = {
    term: re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
    for term in GLOSSARY
}

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

# Report boilerplate stripped before summarization only (NER still sees the
# original full text, since these labels/headers can themselves be useful
# structural cues for entity extraction).
REPORT_TITLE_PATTERN = re.compile(
    r"^\s*SURGICAL PATHOLOGY REPORT\s*$", flags=re.IGNORECASE | re.MULTILINE
)
SECTION_LABELS = [
    "Specimen",
    "Clinical history",
    "Gross description",
    "Microscopic description",
    "Diagnosis",
    "Comment",
]
SECTION_LABEL_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(label) for label in SECTION_LABELS) + r"):\s*",
    flags=re.IGNORECASE | re.MULTILINE,
)

# Placed first in the summarizer input so the diagnosis survives even with a
# short max_length, instead of getting crowded out by earlier sections.
PRIORITY_SECTION_ORDER = ["Diagnosis", "Microscopic description"]


def preprocess_for_summary(report_text: str) -> str:
    """Strip title/labels, keeping content, with diagnosis + microscopic
    findings moved to the front and the remaining sections following."""
    text = REPORT_TITLE_PATTERN.sub("", report_text)

    matches = list(SECTION_LABEL_PATTERN.finditer(text))
    if not matches:
        return text.strip()

    preamble = text[: matches[0].start()].strip()
    sections = []
    for i, match in enumerate(matches):
        label = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((label, text[start:end].strip()))

    priority_keys = [label.lower() for label in PRIORITY_SECTION_ORDER]
    priority = [
        content
        for key in priority_keys
        for label, content in sections
        if label == key
    ]
    remaining = [content for label, content in sections if label not in priority_keys]

    ordered_parts = ([preamble] if preamble else []) + priority + remaining
    return "\n".join(part for part in ordered_parts if part)


DISCLAIMER_MD = (
    "> ⚠️ **Important:** This tool is for **education only**. It is not a "
    "medical device, it can make mistakes, and it does not replace your "
    "doctor. Always discuss your pathology results with your healthcare "
    "provider."
)


# ---------------------------------------------------------------------------
# 3. Core analysis functions
# ---------------------------------------------------------------------------
def highlight_entities(report_text: str):
    """Run biomedical NER and return spans for gr.HighlightedText."""
    entities = ner_pipeline(report_text)

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


def summarize_report(report_text: str) -> str:
    """Generate a short plain-language summary of the report."""
    clean_text = preprocess_for_summary(report_text)

    # The model has an input limit; truncate very long reports defensively.
    words = clean_text.split()
    if len(words) > 400:
        clean_text = " ".join(words[:400])

    n_words = len(clean_text.split())
    inputs = summarizer_tokenizer(clean_text, return_tensors="pt", truncation=True)
    output_ids = summarizer_model.generate(
        **inputs,
        max_length=min(130, max(30, n_words // 2)),  # ~half the input, so it condenses
        min_length=45,
        no_repeat_ngram_size=3,
        do_sample=False,
    )
    return summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def build_glossary_table(report_text: str) -> str:
    """Find glossary terms present in the report and explain them in lay terms."""
    rows = []
    for term, pattern in GLOSSARY_PATTERNS.items():
        if pattern.search(report_text):
            rows.append(f"| **{term.title()}** | {GLOSSARY[term]} |")

    if not rows:
        return (
            "No common pathology terms from the built-in glossary were found "
            "in this report."
        )
    header = "| Term in your report | What it means in plain English |\n|---|---|\n"
    return header + "\n".join(rows)


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

    highlighted = highlight_entities(report_text)
    summary = summarize_report(report_text)
    glossary_md = build_glossary_table(report_text)
    return highlighted, summary, glossary_md


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
# 5. Gradio interface
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")

CUSTOM_CSS = """
#app-title h1 {font-size: 2rem; margin-bottom: 0.2rem;}
#disclaimer {border-left: 4px solid #e67e22; padding-left: 12px;}
"""

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="PathPal - Pathology Report Interpreter") as demo:
    gr.Markdown(
        "# 🔬 PathPal — Pathology Report Interpreter\n"
        "Paste your pathology report (or upload a photo of it) and PathPal will "
        "**highlight the medical terms**, give you a **plain-language summary**, "
        "and **explain common jargon** — so you can walk into your next "
        "appointment with better questions.",
        elem_id="app-title",
    )
    gr.Markdown(DISCLAIMER_MD, elem_id="disclaimer")

    with gr.Tabs():
        # ------------------------- TAB 1: paste text -----------------------
        with gr.Tab("📋 Paste report text"):
            report_input = gr.Textbox(
                label="Pathology report text",
                placeholder="Paste the full text of the pathology report here...",
                lines=12,
            )
            analyze_btn = gr.Button("Explain my report", variant="primary")

        # --------------------- TAB 2: photo upload (OCR) -------------------
        with gr.Tab("📷 Upload a photo of the report (multimodal)"):
            gr.Markdown(
                "Take a clear, well-lit photo of the printed report, straight-on. "
                "The extracted text will appear in the box on the left tab — "
                "review it for OCR errors, then click **Explain my report**."
            )
            image_input = gr.Image(
                label="Photo or scan of the printed report", type="pil"
            )
            ocr_btn = gr.Button("Read text from photo", variant="secondary")

    gr.Markdown("## Results")
    with gr.Row():
        with gr.Column(scale=3):
            highlighted_output = gr.HighlightedText(
                label="Your report with medical terms highlighted",
                combine_adjacent=True,
                show_legend=True,
            )
        with gr.Column(scale=2):
            summary_output = gr.Textbox(
                label="Plain-language summary (AI-generated — may contain errors)",
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
        "[Falconsai/medical_summarization]"
        "(https://huggingface.co/Falconsai/medical_summarization) · "
        "OCR via Tesseract. Built with 🤗 Transformers and Gradio.*"
    )

if __name__ == "__main__":
    demo.launch()
