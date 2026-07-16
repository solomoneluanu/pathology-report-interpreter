---
title: PathPal - Pathology Report Interpreter
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
short_description: Explains pathology reports to patients in plain language
---

# 🔬 PathPal — Pathology Report Interpreter for Patients

**Live demo:** https://huggingface.co/spaces/YOUR_USERNAME/pathology-report-interpreter

Pathology reports are written by doctors, for doctors. Patients are routinely handed
phrases like *"moderately differentiated invasive ductal carcinoma with negative
margins"* and left to guess whether that is good or bad news. PathPal helps patients
understand their own reports so they can ask better questions at their next
appointment.

## What it does

Given the text of a pathology report — pasted directly or extracted from a **photo of
the printed report** (multimodal OCR input) — PathPal produces three things:

1. **Highlighted medical terms.** A biomedical named-entity-recognition (NER) model
   detects diseases, procedures, body structures, lab values, and medications, and
   displays the original report with color-coded highlights.
2. **A plain-language explanation.** A rule-based narrator assembles a short paragraph
   from the detected NER entities and the glossary — naming the specimen/site, the main
   diagnosis, grade/differentiation, and receptor status where present. It is not a
   generative model, so it never states anything beyond what was actually detected in
   the report.
3. **A jargon glossary.** A curated dictionary of 80+ common pathology terms
   (margins, in situ, dysplasia, metastatic, Gleason score, ...) is matched against
   the report and each detected term is explained in plain English.

## Models

| Component | Model | Task |
|---|---|---|
| Term highlighting | [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) | Token classification (biomedical NER), DistilBERT-based |
| Plain-language explanation | Rule-based narrator (`narrate_report` in `app.py`) | Assembles sentences from NER entities + `glossary.json`; no generative model |
| Photo-of-report reading | Tesseract OCR (via `pytesseract`) | Optical character recognition |

The NER model is small enough to run on the free CPU hardware tier of Hugging Face
Spaces.

## How to use it

1. Open the Space and either paste report text into the **Paste report text** tab, or
   switch to the **Upload a photo** tab, upload a clear photo of a printed report, and
   click *Read text from photo*.
2. Click **Explain my report**.
3. Read the highlighted report, the plain-language explanation, and the glossary. Bring
   your questions to your doctor.

Three synthetic example reports (colon polyp, breast biopsy, skin lesion) are built
into the app so the interface can be tested with one click.

## Data sources

- The glossary (`glossary.json`) was hand-curated for this project from publicly
  available patient-education definitions of common pathology terminology.
- The example reports in `sample_inputs/` are **entirely synthetic**. They were written
  for this project to resemble the structure of real reports and contain **no real
  patient data**.

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/pathology-report-interpreter.git
cd pathology-report-interpreter
pip install -r requirements.txt
# OCR feature also requires the Tesseract binary:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
#   Windows:       https://github.com/UB-Mannheim/tesseract/wiki
python app.py
```

The first launch downloads the NER model from the Hugging Face Hub (~250 MB).

## Limitations

- **The plain-language explanation only covers what the NER model and glossary
  actually detect.** If the diagnosis, grade, or receptor status wasn't tagged as an
  entity or matched in the glossary, it won't appear in the explanation — the tool
  favors omission over guessing.
- The NER model was trained on general biomedical text, not specifically on pathology
  reports, so some terms may be missed or mislabeled, which in turn limits what the
  explanation can describe.
- The glossary only covers terms in its curated dictionary; rare or highly specialized
  terminology will not be explained.
- OCR quality depends heavily on photo quality; users are asked to review the
  extracted text before analysis.
- English-language reports only.

## Ethical considerations

- **This is an educational tool, not a medical device.** It must not be used for
  diagnosis or treatment decisions, and the interface displays a persistent
  disclaimer directing users to their healthcare provider.
- **Privacy:** users should remove names, dates of birth, and record numbers before
  pasting a report. The app does not store inputs, but as with any hosted demo, users
  should avoid submitting identifying information.
- **Risk of misinterpretation:** an incomplete or over-simplified explanation of a
  cancer diagnosis could cause real harm (false reassurance or unnecessary alarm). The
  app mitigates this by always showing the original report text alongside the
  explanation, and framing the tool as preparation for a doctor conversation rather
  than a replacement for one.
- All models and libraries used are openly licensed and used within their license
  terms.

## Project structure

```
pathology-report-interpreter/
├── app.py              # main Gradio application
├── requirements.txt    # Python dependencies
├── packages.txt        # system packages for HF Spaces (Tesseract OCR)
├── glossary.json       # curated plain-language glossary (80+ terms)
├── README.md           # this file
├── sample_inputs/      # synthetic example reports
└── assets/             # screenshots
```

## Author

Solomon Tessega — individual course project (Transformer-Based Application using Gradio and
Hugging Face Spaces).
