import re
from pathlib import Path
import yaml
import nltk
from semantic_extractor import extract_semantic

nltk.download('stopwords', quiet=True)



_YAML_PATH = Path(__file__).parent / "symptoms.yaml"

with open(_YAML_PATH, encoding="utf-8") as f:
    _data = yaml.safe_load(f)
    GENERAL_SYMPTOMS = set(_data.pop("_general_symptoms", []))
    SYMPTOM_KEYWORDS = _data

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def phrase_exists(text, phrase):
    pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    return re.search(pattern, text) is not None


def is_negated(text, phrase):
    pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    match = re.search(pattern, text)

    if not match:
        return False

    window = text[max(0, match.start() - 60):match.start()]

    # If there is a contrast word like "but", only check after it
    clause_breaks = [" but ", " however ", " though ", " although ", " yet "]
    for breaker in clause_breaks:
        if breaker in window:
            window = window.split(breaker)[-1]

    negation_patterns = [
        r"\bno\b",
        r"\bnot\b",
        r"\bdont\b",
        r"\bdon t\b",
        r"\bdon't\b",
        r"\bneither\b",
        r"\bnor\b",
        r"\bdo not\b",
        r"\bdoesnt\b",
        r"\bdoes not\b",
        r"\bwithout\b",
        r"\bnever\b"
    ]

    return any(re.search(pattern, window) for pattern in negation_patterns)


def extract_symptoms(text):
    text = clean_text(text)
    detected = set()
    negated = set()

    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        found = False
        neg_seen = False
        for keyword in keywords:
            keyword = clean_text(keyword)
            if not phrase_exists(text, keyword):
                continue
            if is_negated(text, keyword):
                neg_seen = True
            else:
                found = True
                break

        if found:
            detected.add(symptom)
        elif neg_seen:
            negated.add(symptom)

    # semantic fallback: only for symptoms the keyword pass didn't decide on
    sem_found, sem_negated = extract_semantic(text, exclude=detected | negated)
    detected |= (sem_found - sem_negated)
    detected -= sem_negated

    # return in YAML order so output stays deterministic
    return [s for s in SYMPTOM_KEYWORDS if s in detected]


def extract_negated_symptoms(text):
    text = clean_text(text)
    negated = []

    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        all_terms = keywords + [symptom.replace("_", " ")]

        for keyword in all_terms:
            keyword = clean_text(keyword)

            if phrase_exists(text, keyword) and is_negated(text, keyword):
                negated.append(symptom)
                break

        _, sem_negated = extract_semantic(text, exclude=set(negated))
        return [s for s in SYMPTOM_KEYWORDS if s in set(negated) | sem_negated]


def extract_symptoms_from_history(messages):
    active_symptoms = set()

    for message in messages:
        detected = extract_symptoms(message)
        negated = extract_negated_symptoms(message)

        for symptom in detected:
            active_symptoms.add(symptom)

        for symptom in negated:
            active_symptoms.discard(symptom)

    return list(active_symptoms)


def format_symptoms(symptoms):
    return ", ".join(symptom.replace("_", " ") for symptom in symptoms)