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

# 🔬 PathPal — Pathology Report Interpreter

**Live demo:** https://huggingface.co/spaces/solomoneluanu/pathology-report-interpreter

Pathology reports are written by doctors, for doctors. Patients are routinely
handed phrases like *"moderately differentiated invasive ductal carcinoma with
negative margins"* and left to guess whether that is good or bad news. PathPal
helps patients understand their own reports so they can ask better questions at
their next appointment.

---

## What it does

Given the text of a pathology report — pasted directly, or extracted from a
**photo of a printed report** via OCR (multimodal input) — PathPal produces:

1. **Highlighted medical terms.** A biomedical named-entity-recognition (NER)
   model tags diseases, body structures, procedures and lab values, and the
   report is displayed with colour-coded highlights.
2. **A plain-language explanation.** A deterministic narration layer states the
   diagnosis, specimen, grade and receptor status in patient-friendly wording,
   drawing every explanation from a curated glossary.
3. **A jargon glossary.** 86+ common pathology terms (margins, in situ,
   dysplasia, metastatic, Gleason score …) matched against the report and
   explained in plain English.

---

## Why not just paste the report into a general chatbot?

For a single one-off explanation, a large general chatbot will produce more
fluent prose than PathPal, and this project does not claim otherwise. PathPal
exists for the cases where using a general-purpose chatbot is the wrong tool:

- **Explanations cannot be hallucinated.** Every medical definition comes from a
  curated glossary, not a generative model. A generative model is *usually*
  right about "Gleason score" — "usually" is doing dangerous work when the
  subject is a cancer diagnosis.
- **Data control.** The pipeline is small enough to self-host or run offline, so
  patient text need not be sent to a third-party consumer service.
- **Deterministic and auditable.** The same report always yields the same
  output, and every statement traces to either the report text or a glossary
  entry — a requirement in regulated or clinical settings.
- **Bounded scope.** The app does one narrow, safe thing and always routes the
  patient back to their clinician, rather than answering open-ended medical
  questions.
- **Cost at scale.** Batch use across thousands of reports is free to run on
  commodity hardware.

The intended deployment is therefore an institutional one (a clinic or lab
generating patient-friendly explanations), not a replacement for a consumer
chatbot.

---

## Architecture

```
                 ┌───────────────┐
  photo ───────► │  Tesseract    │──┐
  of report      │     OCR       │  │
                 └───────────────┘  │
                                    ▼
  pasted text ───────────────► report text
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      ┌───────────────┐   ┌──────────────────┐   ┌────────────────┐
      │ Biomedical    │   │ Deterministic    │   │ Glossary       │
      │ NER (encoder) │   │ narration        │   │ term matcher   │
      │ → highlights  │   │ → plain English  │   │ → definitions  │
      └───────────────┘   └──────────────────┘   └────────────────┘
```

**Hybrid by design:** the Transformer handles the fuzzy language problem
(recognising entities in messy free text); deterministic code handles the
safety-critical problem (stating what terms mean). There is **no generative
model anywhere in the patient-facing path**, so the explanations cannot invent
medical content.

---

## Models and components

| Component | Model / tool | Task |
|---|---|---|
| Term highlighting | [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) | Token classification (biomedical NER), DistilBERT-based encoder |
| Plain-language narration | rule-based (entities + curated glossary) | Deterministic text generation — no LLM |
| Diagnosis / specimen extraction | rule-based section parser | Guarantees the primary finding is always reported |
| Photo-of-report reading | Tesseract OCR (`pytesseract`) | Optical character recognition (multimodal input) |

The NER model is small enough to run on the free CPU tier of Hugging Face
Spaces.

---

## Evaluation

The NER model's authors benchmarked it on general biomedical text. That number
does not answer this project's question: **does it transfer well enough to
pathology reports specifically?** A domain-transfer evaluation was therefore run
on **15 hand-annotated synthetic pathology reports** (`eval_data.py`,
`evaluation.py`) spanning breast, colon, skin, prostate, lung, thyroid, lymph
node, cervix, stomach, kidney, bladder, liver, endometrium, bone marrow and
pancreas.

**Metrics.** NER is a labelling task, so classification metrics are used —
precision, recall (sensitivity) and F1. Text-overlap metrics such as BLEU
(translation) and ROUGE (summarisation) do not apply, because the model tags
spans rather than generating text.

**Matching.** Entities are compared by **character offsets**, which avoids
artefacts from punctuation and sub-word tokenisation. Two modes are reported:
*strict* (exact span and type) and *lenient* (span overlap and correct type,
crediting boundary near-misses).

| Mode | Precision | Recall (sensitivity) | F1 |
|---|---|---|---|
| Strict (exact span) | 0.083 | 0.086 | 0.085 |
| Lenient (overlap) | 0.278 | 0.286 | 0.282 |

Per entity type:

| Entity type | Mode | Precision | Recall | F1 | n |
|---|---|---|---|---|---|
| Biological_structure | strict | 0.000 | 0.000 | 0.000 | 15 |
| Biological_structure | lenient | 0.160 | 0.267 | 0.200 | 15 |
| Disease_disorder | strict | 0.273 | 0.150 | 0.194 | 20 |
| Disease_disorder | lenient | 0.545 | 0.300 | 0.387 | 20 |

**Findings.**

- **The general biomedical NER model transfers poorly to structured pathology
  reports.** Even with lenient character-overlap matching, disease recall is
  0.300 — the model misses roughly two thirds of diagnoses in this register of
  text.
- **Multi-token diagnoses are fragmented into sub-word pieces.** On the breast
  report, `invasive ductal carcinoma` was returned as the fragments `duct` and
  `##cino`, with the tokenizer artefact leaking into the output; the descriptor
  `moderately` was tagged as an entity, and the specimen organ `breast` was not
  tagged at all.
- **Likely cause:** the model was trained on flowing biomedical prose, whereas
  surgical pathology reports use a terse, templated register (`Specimen: Left
  breast, core needle biopsy.`) that falls outside its training distribution.
- Specialised pathology **procedure** terms (polypectomy, lobectomy, Whipple
  resection) were handled too inconsistently to annotate fairly — `Whipple` is
  split into `whip` / `##ple` and labelled `Detailed_description` /
  `Sign_symptom` — and were therefore excluded from the scored entity set. This
  scoping decision is documented rather than hidden.

**How this shaped the design.** This measurement is the empirical justification
for PathPal's architecture. Because NER cannot be trusted on this text:

- the **diagnosis and specimen are parsed deterministically** from the labelled
  report sections, so the primary finding is always surfaced regardless of model
  performance (a live test in which `basal cell carcinoma` went untagged would
  otherwise have omitted the diagnosis entirely);
- all **patient-facing explanations come from the curated glossary**, never from
  a model;
- **NER is retained only for the supplementary highlighting layer**, where
  partial detection is acceptable — a missed highlight is cosmetic, whereas a
  missed diagnosis would be harmful.

Gold annotations were produced by a single annotator (the author) — appropriate
for a course project; a publication would require multiple annotators and an
inter-annotator agreement measure.

---

## Limitations

- **Sub-word fragmentation.** Multi-token entities are sometimes split, which
  affects highlight boundaries.
- **No negation detection.** Terms mentioned in the negative ("*no* evidence of
  invasive carcinoma") are still surfaced in the glossary, because the tool
  detects term presence rather than assertion status.
- **Glossary coverage is finite.** Rare or highly specialised terminology
  outside the curated glossary is not explained.
- **Narration is template-based**, so it is safe and consistent but not as
  fluent as a large generative model would be.
- **OCR quality depends on photo quality**; users are asked to review the
  extracted text before analysis.
- **English-language reports only.**

---

## Ethical considerations

- **Educational tool, not a medical device.** It must not be used for diagnosis
  or treatment decisions. A persistent disclaimer directs users to their
  healthcare provider.
- **Privacy.** The application does not store inputs; models run inference only,
  so user text is never used for training. Users are nevertheless advised to
  remove names, dates of birth and record numbers before pasting a report, as
  with any hosted demo.
- **Risk of misinterpretation.** A wrong or over-simplified summary of a cancer
  diagnosis could cause real harm. This is mitigated by always displaying the
  original report alongside the explanation, sourcing all definitions from a
  curated glossary, and framing the tool as preparation for a clinician
  conversation rather than a substitute for one.
- **Synthetic data only.** All example reports were written for this project and
  contain no real patient data.
- All models and libraries are openly licensed and used within their terms.

---

## Running locally

```bash
git clone https://github.com/solomoneluanu/pathology-report-interpreter.git
cd pathology-report-interpreter
pip install -r requirements.txt
python app.py
```

The OCR feature additionally requires the Tesseract binary:
Ubuntu/Debian `sudo apt-get install tesseract-ocr`, macOS `brew install
tesseract`, Windows via the UB-Mannheim installer. On Hugging Face Spaces this
is installed automatically from `packages.txt`.

To reproduce the evaluation:

```bash
python evaluation.py
```

---

## Project structure

```
pathology-report-interpreter/
├── app.py              # Gradio application
├── evaluation.py       # NER domain-transfer evaluation
├── eval_data.py        # 15 annotated synthetic reports (gold standard)
├── glossary.json       # curated plain-language glossary (86+ terms)
├── requirements.txt    # Python dependencies
├── packages.txt        # system packages for Spaces (Tesseract OCR)
├── README.md
├── sample_inputs/      # synthetic example reports
└── assets/             # screenshots
```

---

## Author

**Solomon Tessega**

Individual course project — Transformer-Based Application using Gradio and
Hugging Face Spaces. All implementation and writing are my own work.
