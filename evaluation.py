"""
evaluation.py - NER domain-transfer evaluation for PathPal
==========================================================

Measures how well the off-the-shelf biomedical NER model
(d4data/biomedical-ner-all) detects entities WHEN APPLIED TO PATHOLOGY REPORTS.

This is a DOMAIN-TRANSFER evaluation: the model's authors already benchmarked it
on general biomedical text; the question here is whether it transfers to
pathology reports well enough to power PathPal.

METRICS (and why not BLEU/ROUGE):
  NER is a labeling task, so classification metrics are used. BLEU (translation)
  and ROUGE (summarization) compare generated TEXT to a reference and do not
  apply to span labeling.
    - Precision   = of entities flagged, fraction correct
    - Recall      = of real entities, fraction found  (a.k.a. SENSITIVITY)
    - F1          = harmonic mean of precision and recall

MATCHING (character-offset based, robust to sub-word tokenization):
  Earlier whitespace-token matching mis-scored correct predictions because of
  trailing punctuation ("breast," vs "breast") and sub-word fragmentation. This
  version compares CHARACTER spans directly, which is the standard robust way to
  evaluate NER independent of tokenizer quirks.

  Two modes reported:
    STRICT  - predicted and gold character spans must match exactly (same type).
    LENIENT - a gold entity counts as found if a predicted span of the same type
              OVERLAPS it (credits boundary/sub-word near-misses).

DATA:
  15 synthetic, hand-annotated pathology reports (eval_data.py). No real patient
  data. Single-annotator gold labels - fine for a course project; a publication
  would use multiple annotators + agreement.

USAGE:
  pip install seqeval        # (not required anymore, but harmless)
  python evaluation.py
"""

import re
from transformers import pipeline
from eval_data import GOLD_DATA


def gold_spans(text, entities):
    """Return list of (start_char, end_char, type) for gold entities."""
    spans = []
    for entity_text, etype in entities:
        for m in re.finditer(re.escape(entity_text), text):
            spans.append((m.start(), m.end(), etype))
            break  # first occurrence only
    return spans


def pred_spans(ner_results, keep_types):
    """Return list of (start_char, end_char, type) for predicted entities."""
    spans = []
    for ent in ner_results:
        etype = ent["entity_group"]
        if etype in keep_types:
            spans.append((ent["start"], ent["end"], etype))
    return spans


def overlaps(a, b):
    """Do two character spans overlap? a,b = (start,end,type)."""
    return a[0] < b[1] and b[0] < a[1]


def score(gold, pred, strict):
    """Count TP/FP/FN comparing gold vs pred character spans."""
    matched_pred = set()
    tp = 0
    for g in gold:
        hit = False
        for j, p in enumerate(pred):
            if j in matched_pred:
                continue
            if p[2] != g[2]:
                continue
            same = (p[0] == g[0] and p[1] == g[1]) if strict else overlaps(g, p)
            if same:
                hit = True
                matched_pred.add(j)
                break
        if hit:
            tp += 1
    fn = len(gold) - tp
    fp = len(pred) - len(matched_pred)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    print("Loading NER model...\n")
    ner = pipeline(
        "token-classification",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple",
    )

    keep_types = {et for item in GOLD_DATA for _, et in item["entities"]}

    # accumulate spans across all reports, tagged by type for per-type scoring
    all_gold, all_pred = [], []
    for item in GOLD_DATA:
        text = item["text"]
        g = gold_spans(text, item["entities"])
        p = pred_spans(ner(text), keep_types)
        all_gold.append(g)
        all_pred.append(p)

    def aggregate(strict, type_filter=None):
        TP = FP = FN = 0
        for g, p in zip(all_gold, all_pred):
            if type_filter:
                g = [s for s in g if s[2] == type_filter]
                p = [s for s in p if s[2] == type_filter]
            tp, fp, fn = score(g, p, strict)
            TP += tp; FP += fp; FN += fn
        return TP, FP, FN

    print("=" * 62)
    print("NER DOMAIN-TRANSFER EVALUATION - pathology reports")
    print("=" * 62)
    print(f"Reports: {len(GOLD_DATA)}   Types: {sorted(keep_types)}\n")

    for strict in (True, False):
        label = "STRICT (exact char span + type)" if strict else \
                "LENIENT (character overlap + type)"
        TP, FP, FN = aggregate(strict)
        p, r, f = prf(TP, FP, FN)
        print(f"--- {label} ---")
        print(f"Precision {p:.3f}   Recall/Sensitivity {r:.3f}   F1 {f:.3f}"
              f"   (TP={TP} FP={FP} FN={FN})")
        for t in sorted(keep_types):
            tp, fp, fn = aggregate(strict, t)
            pp, rr, ff = prf(tp, fp, fn)
            print(f"    {t:22s} P {pp:.3f}  R {rr:.3f}  F1 {ff:.3f}  (n={tp+fn})")
        print()

    print("Interpretation:")
    print("- Character-offset matching removes punctuation / sub-word artifacts,")
    print("  so these numbers reflect real detection ability.")
    print("- The strict->lenient gain shows how often the model finds an entity")
    print("  but with slightly different boundaries.")
    print("- Disease_disorder recall is the safety-critical number: any residual")
    print("  miss is why PathPal extracts the diagnosis deterministically rather")
    print("  than trusting NER for it.")


if __name__ == "__main__":
    main()
