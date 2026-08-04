import pandas as pd
import random
import gradio as gr
import re
from nltk.corpus import stopwords # type: ignore
import joblib
from pathlib import Path
import yaml
from symptom_extractor import extract_symptoms_from_history, format_symptoms, GENERAL_SYMPTOMS

_DIR = Path(__file__).parent
with open(_DIR / "diseases.yaml", encoding="utf-8") as f:
    disease_advice = yaml.safe_load(f)


symptom_model = joblib.load("symptom_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# Instructions and UI setup
instructions = """<h2>Welcome to MediChat! Your Medical Chatbot</h2>
<ol>
  <li><b>How to Start:</b> Type your symptoms in the textbox and press enter.</li>
  <li>The bot will respond with the best possible answers to your messages. For now, let's keep it simple as we continue to enhance its capabilities.</li>
  <p><b>Disclaimer:</b> This chatbot is for educational purposes only and does not provide medical diagnosis. Please consult a qualified healthcare professional for medical concerns.</p>
</ol> 
  
"""

# Basic function to check if the input is gibberish
def is_gibberish(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    if len(words) < 2 or all(word.lower() in stop_words for word in words):
        return True
    return False

def clean_symptom_input(text):
    #normalize input for matching
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).strip().lower()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  
    return cleaned_text

def recognize_greetings(message):
    greetings = ["hello", "hi","salam","hiya", "hey", "good morning","good afternoon", "good noon", "hi!", "hola!",  "good evening"]
    goodbyes = ["bye", "thank you","bubye","goodnight","cheers","see you later","see ya", "goodbye", "take care", "see you"]
    small_talks = [
        "how are you", "how r you", "how are ya", "how you doing", "how u doing", "how do you do", "howdy", "hey how are you", "hi how r u",
        "what's up", "sup", "wassup", "wazzup", "yo", "hey there", "hiya", "how goes it", "how's it going", "how's things", "how's life",
        "how's everything", "how are things", "how are you getting on", "how are you keeping", "you alright", "you good", "u good", "all good",
        "how have you been", "how've you been", "how u been", "long time no see", "where have you been", "what's good", "what's cracking",
        "what's happening", "what's new", "what's the good word", "what's the scoop", "what's the story", "what's the deal",
        "what's new with you", "what's new w u", "what's up lately", "what have you been up to", "what've you been up to", 
        "what u been up to", "what's going on", "what's going on with you", "what's happening with you", "been busy", "keeping busy", 
        "what are you up to", "what r u up to", "what are you doing", "whatcha doing", "what's keeping you busy", "anything new", "any updates", 
        "give me the news", "catch me up", "fill me in", "what's been going on", "what's the latest", "what's new in your world", 
        "how's your world", "what's shaking", "what's the latest gossip", "got any plans tonight", "got plans tonight", "what are you doing later", 
        "what u doing later", "what's on for tonight", "anything on tonight", "busy later", "free later", "got time to chat", "what are your plans for the future", 
        "where do you see yourself in 5 years", "any trips coming up", "going anywhere nice", "planning any travel", "holidays soon", 
        "got any days off coming", "taking time off soon", "how are you feeling", "how r u feeling", "you okay", "u ok", "you alright", "feeling better", 
        "you getting over that cold", "taken your meds", "feeling any better", "how's your health", "been sleeping okay", "eating well", "staying healthy", 
        "how's your back", "how's your head", "headache gone", "feeling fit", "you seem tired, you ok", "you look tired, everything alright",
        "how are you feeling today", "you in a good mood", "feeling positive", "having a good day", "struggling today", "hanging in there", 
        "you look happy","hi how are you", "hello how are you", "hey how are you", "hi how you doing", "hello how you doing", "you seem down, you ok", "you seem off, everything fine", "you doing alright", "you holding up okay", "keeping your head up", 
        "staying strong", "you sound tired", "you sound stressed, need to vent"
        ]
    
    message_cleaned = clean_symptom_input(message)
    

    if message_cleaned in greetings:
        return "greeting"
    elif message_cleaned in goodbyes:
        return "goodbye"
    elif message_cleaned in small_talks:
        return "smalltalk"

    return None

def respond(message, chat_history, symptoms_state, questions_state, feedback_state, attempt_state):
    # Unpack state
    user_symptoms = symptoms_state or []
    questions_asked = questions_state or []
    feedback_given = feedback_state
    attempts = attempt_state or 0

    # Clean and preprocess message
    cleaned_message = clean_symptom_input(message)
    if cleaned_message in ["reset", "start over", "new case", "clear symptoms"]:
        bot_message = "Okay, I have cleared the previous symptoms. Please describe the new symptoms."
        chat_history.append((message, bot_message))
        return "", chat_history, [], [], False, 0

    # Check for greetings and goodbyes
    response_type = recognize_greetings(message)
    
    if response_type == "greeting":
        bot_message = random.choice([
            "Hello! I'm here to help. Please describe any symptoms you're experiencing.",
            "Hi there! What symptoms are you facing today?",
            "Greetings! Let me know what you're feeling, and I'll assist you."
        ])
        chat_history.append((message, bot_message))
        return "", chat_history, user_symptoms, questions_asked, feedback_state, attempts
    elif response_type == "goodbye":
        bot_message = random.choice([
            "Take care! If you have more questions, feel free to ask.",
            "Goodbye! Stay healthy, and reach out if you need more help."
        ])
        chat_history.append((message, bot_message))
        return "", chat_history, [], [], False, 0
    elif response_type == "smalltalk":
        bot_message = random.choice([
            "I'm medical symptoms categorization chatbot and I'm ready to help. Please tell me your symptoms.",
            "I am not sure what are you talking about. Could you describe your symptoms so I can assist you?",
            "I can only take symptoms as input. Could you describe your symptoms so I can assist you?",
            "I can assist with symptoms. Please describe your symptoms so I can assist you."

        ])
        chat_history.append((message, bot_message))
        return "", chat_history, [], [], False, 0

    # Handle gibberish input
    quick_symptoms = extract_symptoms_from_history([cleaned_message])

    if not quick_symptoms and is_gibberish(cleaned_message):
        bot_message = "I couldn't quite understand that. Could you describe your symptoms in more detail?"
        chat_history.append((message, bot_message))
        return "", chat_history, user_symptoms, questions_asked, feedback_state, attempts

    # Process symptoms
    user_symptoms.append(cleaned_message)
    combined_symptoms = " ".join(user_symptoms)
    detected_symptoms = extract_symptoms_from_history(user_symptoms)

    if len(detected_symptoms) >= 2:

        # Make prediction
        try:
            user_row = {
                symptom: 1 if symptom in detected_symptoms else 0
                for symptom in symptom_columns
            }

            X_user = pd.DataFrame([user_row])

            probabilities = symptom_model.predict_proba(X_user)[0]
            class_labels = symptom_model.classes_

            top_indices = probabilities.argsort()[-3:][::-1]

            top_matches = [
                (class_labels[i], probabilities[i])
                for i in top_indices
            ]

            top_disease, top_score = top_matches[0]
            second_disease, second_score = top_matches[1]
            score_gap = top_score - second_score

            detected_text = format_symptoms(detected_symptoms)

            specific_symptoms = [
                symptom for symptom in detected_symptoms
                if symptom not in GENERAL_SYMPTOMS
            ]

            general_only = len(specific_symptoms) == 0

            if len(detected_symptoms) < 3:
                bot_message = (
                    f"I detected these symptoms: {detected_text}. "
                    "These symptoms are too general to suggest one condition safely. "
                    "Please add more details such as duration, headache, stomach pain, diarrhoea, vomiting, chills, rash, cough, body pain, or pain location. "
                    "This is not a diagnosis."
                )

            elif general_only and len(detected_symptoms) < 4:
                bot_message = (
                    f"I detected these symptoms: {detected_text}. "
                    "These symptoms can appear in many conditions, so I need more detail before suggesting a likely match. "
                    "Please add any more specific symptoms such as stomach pain, diarrhoea, vomiting, chills, rash, cough, sore throat, burning urination, or pain location. "
                    "This is not a diagnosis."
                )

            elif general_only and (top_score < 0.75 or score_gap < 0.25):
                bot_message = (
                    f"I detected these symptoms: {detected_text}. "
                    "These symptoms are still quite general, and the model is not strong enough to suggest one condition safely. "
                    "Top model matches are: "
                    f"{top_matches[0][0]} ({top_matches[0][1]*100:.1f}% model score), "
                    f"{top_matches[1][0]} ({top_matches[1][1]*100:.1f}% model score), "
                    f"and {top_matches[2][0]} ({top_matches[2][1]*100:.1f}% model score). "
                    "Please add more specific details before relying on a match. "
                    "This is not a diagnosis."
                )

            elif top_score < 0.50 or score_gap < 0.10:
                bot_message = (
                    f"I detected these symptoms: {detected_text}. "
                    "The symptom combination does not strongly match one condition. "
                    "Top model matches are: "
                    f"{top_matches[0][0]} ({top_matches[0][1]*100:.1f}% model score), "
                    f"{top_matches[1][0]} ({top_matches[1][1]*100:.1f}% model score), "
                    f"and {top_matches[2][0]} ({top_matches[2][1]*100:.1f}% model score). "
                    "Please add more details such as duration, fever pattern, rash, vomiting, cough, pain location, or breathing issues. "
                    "This is not a diagnosis."
                )

            else:
                bot_message = (
                    f"I detected these symptoms: {detected_text}. "
                    f"The strongest model match is: {top_disease} "
                    f"with {top_score*100:.1f}% model score. "
                    f"General advice: {disease_advice.get(top_disease, 'Please consult a healthcare professional.')} "
                    "This is not a diagnosis. Please consult a healthcare professional for proper medical advice."
                )
                    # user_symptoms = []
        except Exception as e:
            print(f"Error: {e}")
            bot_message = "Something went wrong processing that. Please try rephrasing your symptoms."
    
    else:
        bot_message = (
            "Please describe your symptoms in more detail. "
            "For example, mention symptoms, duration, pain location, fever, rash, vomiting, cough, or sleep issues."
        )
    
    chat_history.append((message, bot_message))
    return "", chat_history, user_symptoms, questions_asked, feedback_state, attempts

# Gradio UI definition
with gr.Blocks() as demo:
    gr.HTML('<h1 align="center">Medical Chatbot: Your Virtual Health Guide</h1>')
    gr.HTML(instructions)

    medi_chat = gr.Chatbot(label="MediChat")
    msg = gr.Textbox(label="Enter your symptoms here")
    # clear = gr.ClearButton([msg, medi_chat], value="Clear Chat")
    clear = gr.Button("Clear Chat")


    # Define states for user symptoms, questions asked, feedback given, and attempts
    symptoms_state = gr.State([])
    questions_state = gr.State([])
    feedback_state = gr.State(False)
    attempt_state = gr.State(0)

    msg.submit(respond, [msg, medi_chat, symptoms_state, questions_state, feedback_state, attempt_state], 
               [msg, medi_chat, symptoms_state, questions_state, feedback_state, attempt_state])
    clear.click(
        lambda: ("", [], [], [], False, 0),
        outputs=[msg, medi_chat, symptoms_state, questions_state, feedback_state, attempt_state]
    )
demo.launch()
