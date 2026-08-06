"""
Semantic fallback for symptom extraction.

The keyword matcher in symptom_extractor.py is precise but only fires on
phrasings that were written down by hand. This module catches the rest:
clauses that clearly describe a symptom but use wording nobody listed.

Used as a fallback, not a replacement - keyword matches always win.
"""

import re
from pathlib import Path
import yaml

_MODEL_NAME = "all-MiniLM-L6-v2"
_THRESHOLD = 0.55          # tune this - see notes at the bottom
_model = None
_canon_names = None
_canon_emb = None


def _load_model():
    """Loaded lazily so importing this module stays cheap."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _canonical_phrases():
    """
    Every phrasing from symptoms.yaml, each mapped back to its symptom.
    Comparing against all variants rather than one gives the model far more
    surface to match against - "silvery flakes" is close to some skin
    phrasings even if it is far from the first one listed.
    """
    path = Path(__file__).parent / "symptoms.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.pop("_general_symptoms", None)

    names, phrases = [], []
    for symptom, variants in data.items():
        for variant in variants or []:
            names.append(symptom)
            phrases.append(variant)
    return names, phrases


def _build_index():
    global _canon_names, _canon_emb
    if _canon_emb is None:
        model = _load_model()
        _canon_names, phrases = _canonical_phrases()
        _canon_emb = model.encode(phrases, convert_to_tensor=True,
                                  normalize_embeddings=True)
    return _canon_names, _canon_emb


_CLAUSE_SPLIT = re.compile(r"[,;.]| and | but | however | though | also ")


def split_clauses(text):
    """
    Break a message into short spans. Embedding a whole sentence washes out
    individual symptoms; embedding clauses keeps each one distinct.
    """
    parts = _CLAUSE_SPLIT.split(text.lower())
    return [p.strip() for p in parts if len(p.strip()) > 2]


_NEGATION_CUES = (
    "no ", "not ", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "haven't", "havent", "hasn't", "hasnt", "without", "free of", "denies",
    "never had", "ruled out",
    "is fine", "are fine", "is normal", "are normal", "is okay", "is ok",
    "feels fine", "feels normal", "nothing wrong",
)


def is_negated_clause(clause):
    """
    Clause-level negation. Simpler and more reliable than scanning backwards
    from a matched phrase, because the clause boundary already scopes it.
    """
    return any(cue in clause for cue in _NEGATION_CUES)


def extract_semantic(text, threshold=_THRESHOLD, exclude=None):
    """
    Return symptoms found by similarity that the keyword matcher missed.

    exclude: symptoms already found by keyword matching, so we don't
             re-detect them or fight with their negation handling.
    """
    from sentence_transformers import util

    exclude = exclude or set()
    names, canon_emb = _build_index()
    model = _load_model()

    clauses = split_clauses(text)
    if not clauses:
        return set(), set()

    clause_emb = model.encode(clauses, convert_to_tensor=True,
                              normalize_embeddings=True)
    scores = util.cos_sim(clause_emb, canon_emb)

    found, negated = set(), set()
    for i, clause in enumerate(clauses):
        row = scores[i]
        best = int(row.argmax())
        if float(row[best]) < threshold:
            continue
        symptom = names[best]
        if symptom in exclude:
            continue
        if is_negated_clause(clause):
            negated.add(symptom)
        else:
            found.add(symptom)

    return found, negated


def explain(text, top_k=3, threshold=0.0):
    """
    Debug helper: show what each clause matched and how strongly.
    Use this to pick a threshold rather than guessing.
    """
    from sentence_transformers import util

    names, canon_emb = _build_index()
    model = _load_model()
    clauses = split_clauses(text)
    if not clauses:
        return []

    clause_emb = model.encode(clauses, convert_to_tensor=True,
                              normalize_embeddings=True)
    scores = util.cos_sim(clause_emb, canon_emb)

    out = []
    for i, clause in enumerate(clauses):
        row = scores[i]
        order = row.argsort(descending=True)[:top_k]
        # matches = [(names[int(j)], round(float(row[int(j)]), 3)) for j in order]
        seen, matches = set(), []
        for j in order:
            n = names[int(j)]
            if n in seen:
                continue
            seen.add(n)
            matches.append((n, round(float(row[int(j)]), 3)))
            if len(matches) >= top_k:
                break
        out.append((clause, [m for m in matches if m[1] >= threshold]))
    return out


if __name__ == "__main__":
    samples = [
    "I don't have a fever but I do have a rash",
    "no cough, no sore throat",
    "my skin is fine, no itching",
    "I have a headache and I am not vomiting",
    "never had chest pain",
    ]
    for s in samples:
        found, negated = extract_semantic(s)
        print(f"\n{s!r}")
        print("   detected:", sorted(found))
        print("   negated :", sorted(negated))
    # for s in samples:
    #     print(f"\n{s!r}")
    #     for clause, matches in explain(s, top_k=3):
    #         print(f"   {clause!r} -> {matches}")