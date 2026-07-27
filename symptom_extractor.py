import re

SYMPTOM_KEYWORDS = {
    "fever": ["fever", "high temperature", "temperature", "hot body"],
    "chills": ["chills", "cold at night", "feeling cold", "shivering", "cold body"],
    "headache": ["headache","headaches", "head pain", "head hurts", "heavy head", "heavy headed"],
    "nausea": ["nausea", "feel sick", "feeling sick", "sick feeling"],
    "vomiting": ["vomiting", "vomit", "puking", "puke", "throwing up", "feel like vomiting", "feel like puking"],
    "fatigue": ["fatigue", "tired", "weak", "lazy", "worn out", "exhausted", "low energy"],
    "rash": ["rash", "red spots", "spots", "skin spots", "red bumps", "bumps", "skin bumps", "red lesions", "lesions", "skin lesions",
             "red patches", "patches", "skin patches", "red marks", "marks", "skin marks",
             "red welts", "welts", "skin welts", "red hives", "hives", "skin hives","skin redness", "red skin", "red patches", "patchy skin", "skin inflammation", "inflamed skin", "skin irritation"],
    "itching": ["itching", "itchy", "scratch", "scratching", "skin itch", "skin is itchy", "itchy skin", "itchy rash", "rash is itchy", "itchy spots", "spots are itchy"
                "itchy bumps", "bumps are itchy", "itchy lesions", "lesions are itchy", "itchy patches", "patches are itchy"],
    "blocked_nose": ["blocked nose", "stuffy nose", "nose blocked", "nasal congestion"],
    "runny_nose": ["runny nose" , "running nose", "nose running", "nasal discharge", "nasal drip", "nose discharge", "nose drip", "nose is running"],
    "cough": ["cough", "coughing"],
    "sore_throat": ["sore throat", "throat pain", "pain in throat"],
    "breathing_difficulty": ["breathing difficulty", "difficulty breathing", "can't breathe", "cannot breathe", "breath normally"],
    "chest_pain": ["chest pain", "pain in chest"],
    "stomach_pain": ["stomach pain", "belly pain", "abdominal pain", "pain in stomach"],
    "diarrhea": ["diarrhea", "loose motion", "loose motions"],
    "constipation": ["constipation", "hard stool"],
    "burning_urination": ["burning urination", "burning pee", "burning when urinating", "painful urination",
                         "painful pee", "pain when urinating", "painful urination", "painful pee", "pain when urinating",
                         "burning sensation while urinating", "burning sensation while peeing", "pain when peeing"],
    "frequent_urination": ["frequent urination", "urinating often", "pee often", "urinating frequently", "pee frequently", "frequent urination", "frequent peeing",
                           "frequent urination", "frequent peeing", "urinating frequently", "urinating often", "pee often", "pee frequently",
                           "more frequent urination", "more frequent peeing", "urinating more often", "peeing more often", "urinating more frequently", "peeing more frequently"],
    "yellow_skin": ["yellow skin", "yellowish skin", "skin yellow"],
    "yellow_eyes": ["yellow eyes", "eyes yellow"],
    "joint_pain": ["joint pain", "pain in joints", "joints hurt"],
    "muscle_pain": ["muscle pain", "body pain", "body ache", "body hurts", "muscle hurts"],
    "back_pain": ["back pain", "pain in back"],
    "neck_pain": ["neck pain", "pain in neck"],
    "eye_pain": ["eye pain", "pain behind eyes", "pain behind my eyes", "eye socket", "eye sockets"],
    "dizziness": ["dizzy", "dizziness", "light headed", "lightheaded"],
    "poor_sleep": ["can't sleep", "cannot sleep", "sleep problem", "poor sleep", "not sleeping"],
    "skin_peeling": ["skin peeling", "peeling skin"],
    "swelling": ["swelling", "swollen"],
    "weight_loss": ["weight loss", "losing weight"],
    "loss_of_appetite": [
    "loss of appetite", "lost appetite", "lost my appetite",
    "poor appetite", "reduced appetite",
    "cant eat", "can't eat", "cannot eat",
    "not eating", "dont like eating", "don't like eating",
    "do not like eating", "dont like eating food",
    "don't like eating food", "dont feel like eating",
    "apetite", "appetite"
    ],
    "acidity": ["acidity", "acid reflux", "heartburn"],
    "sore_throat": [
    "sore throat", "throat pain", "pain in throat",
    "throat irritation", "scratchy throat", "itchy throat",
    "something in my throat", "something in throat",
    "throat discomfort", "phlegm in throat", "mucus in throat"
    ],
    "indigestion": ["indigestion", "digestive problem"],
    "sneezing": ["sneezing", "sneeze"],
    "red_eyes": ["red eyes", "eye redness"],
    "skin_blisters": ["blisters", "skin blisters"],
    "pus": ["pus", "pus filled"],
    "dark_urine": ["dark urine"],
    "sweating": ["sweating", "sweat"],
    "dehydration": ["dehydration", "dehydrated"],
    "leg_swelling": [
    "leg swelling", "swelling in legs", "swelling in my legs",
    "swollen legs", "swollen leg", "legs are swollen"
    ],

    "leg_pain": [
        "leg pain", "pain in legs", "pain in my legs",
        "legs pain", "legs hurt", "painful legs"
    ],
      
    "high_blood_pressure": [
            "high blood pressure", "blood pressure high", "hypertension",
            "high bp", "bp high", "elevated blood pressure", "blood pressure elevated",
            "high blood pressure readings", "blood pressure readings high",
            "high blood pressure levels", "blood pressure levels high"
        ],
    "high_sugar":[
            "high sugar", "sugar high", "elevated sugar", "blood sugar high",
            "high blood sugar", "blood sugar elevated", "high sugar levels", "sugar levels high",
            "high sugar readings", "sugar readings high", "high sugar readings", "blood sugar readings high"
        ],
    "thirst": [
            "thirst", "thirsty", "feeling thirsty", "very thirsty",
            "extremely thirsty", "dehydrated and thirsty", "parched",
            "dry mouth", "mouth is dry", "thirsty and dehydrated"
        ],
    "stiffness": [
            "stiffness", "stiff", "stiff joints", "stiff muscles", "stiff body",
            "stiffness in joints", "stiffness in muscles", "stiffness in body", "stiff and sore", "stiff and achy"
        ],
    "red_sores": [
            "red sores", "sores", "sores on skin", "skin sores",
            "red spots", "spots on skin", "skin spots", "red bumps", "bumps on skin",
            "skin bumps", "red lesions", "lesions on skin", "skin lesions"
        ],

    "pimples": [
            "pimples","zits", "breakouts", "pimple breakouts", "acne breakouts", "zits on skin", "breakouts on skin"
        ],
    "acne": [
            "acne", "acne breakouts", "acne on skin", "skin acne"
        ],
    "scaling_patches": [
            "scaling patches", "patches on skin", "skin patches",
            "scaly skin", "scaly patches", "patchy skin", "patches of skin"
        ],
    "itchy_scalp": [
            "itchy scalp", "scalp itch", "scalp is itchy", "itchy head", "head itch", "head is itchy",
            "itchy hair", "hair itch", "hair is itchy"
        ],
    "pain_while_passing_stool": [
            "pain while passing stool", "pain during bowel movement",
            "pain while defecating", "pain during defecation", "pain while pooping",
            "pain during pooping", "pain while having a bowel movement",
            "pain during having a bowel movement", "pain while having a bowel movement",
        ],
    "swelling_near_anus": [
            "swelling near anus", "swelling around anus", "swelling near rectum",
            "swelling around rectum", "swelling near buttocks", "swelling around buttocks",
            "swelling near anal area", "swelling around anal area", "swelling near rectal area", "swelling around rectal area"
        ],
    "medicine_reaction": [
            "reaction to medicine", "medicine reaction", "adverse reaction to medicine",
            "adverse medicine reaction", "side effect of medicine", "medicine side effect",
            "allergic reaction to medicine", "allergic medicine reaction", "medicine allergy"
        ],
    "low_blood_pressure": [
            "low blood pressure", "blood pressure low", "hypotension",
            "low bp", "bp low", "elevated blood pressure", "blood pressure elevated",
            "low blood pressure readings", "blood pressure readings low",
            "low blood pressure levels", "blood pressure levels low",

        ],

    "calf_pain": [
        "calf pain", "pain in calves", "pain in my calves",
        "calves pain", "calves hurt"
    ],

    "protruding_veins": [
        "protruding veins", "veins protruding", "bulging veins",
        "visible veins", "noticeable veins", "veins are visible",
        "veins in legs", "leg veins", "varicose veins"
    ],

    "vein_swelling": [
        "vein swelling", "veins swelling", "swollen veins",
        "swollen vein", "veins are swollen"
    ],

    "inflamed_skin": [
        "inflamed skin", "skin inflamed", "skin is inflamed",
        "itchy and inflamed", "red and inflamed"
    ],

    "difficulty_walking": [
        "difficulty walking", "difficult to walk",
        "hard to walk", "cannot walk properly",
        "can't walk properly"
    ],
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def phrase_exists(text, phrase):
    pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    return re.search(pattern, text) is not None


def is_negated(text, phrase):
    match = re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text)

    if not match:
        return False

    window = text[max(0, match.start() - 45):match.start()]

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
    detected = []

    for symptom, keywords in SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            keyword = clean_text(keyword)

            if phrase_exists(text, keyword) and not is_negated(text, keyword):
                detected.append(symptom)
                break

    return detected


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

    return negated


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