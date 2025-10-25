# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import os
import re

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and vectorizer (ensure train_model.py has been run)
MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("model.joblib or vectorizer.joblib missing. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
vec = joblib.load(VECTORIZER_PATH)

SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "body_pain",
    "rash", "sneezing", "nausea", "vomiting", "joint_pain", "sore_throat", "diarrhea"
]

PRECAUTIONS = {
    "Common Cold": ["Rest", "Stay hydrated", "Gargle with warm salt water", "Over-the-counter decongestants if needed"],
    "Flu": ["See doctor for severe symptoms", "Rest & hydrate", "Paracetamol for fever (doctor advice)"],
    "Dengue": ["Seek medical help urgently for high fever", "Avoid NSAIDs like ibuprofen", "Stay hydrated"],
    "Malaria": ["Get blood test urgently", "Treat with antimalarial if positive", "Use mosquito protection"],
    "Allergy": ["Avoid allergen if known", "Antihistamines (doctor advice)", "Nasal spray if prescribed"],
    "Gastroenteritis": ["Oral rehydration", "Avoid heavy food until vomiting stops", "Seek care if dehydrated"],
    "Strep Throat": ["See doctor for antibiotics if bacterial", "Gargle and rest", "Pain relief as recommended"]
}

# Simple intent/responses for general health chat (demo)
GENERIC_RESPONSES = [
    "Can you please describe your symptoms more clearly? (e.g., fever, cough, how long you’ve had them?)",
    "If you are having trouble breathing or very high fever, please consult a doctor immediately.",
    "I am a demo assistant. For serious symptoms, always contact a medical professional."
]

def extract_symptoms_from_text(text):
    text = text.lower()
    found = []
    for s in SYMPTOMS:
        if s.replace('_',' ') in text or s in text:
            if re.search(r'\b' + re.escape(s.replace('_',' ')) + r'\b', text) or re.search(r'\b' + re.escape(s) + r'\b', text):
                found.append(s)
    return list(set(found))

@app.route("/")
def index():
    return render_template("index.html", symptoms=SYMPTOMS)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    explicit_symptoms = data.get("symptoms", None)

    # 1) If user provided explicit list
    if explicit_symptoms:
        selected = [s for s in explicit_symptoms if s in SYMPTOMS]
    else:
        # Try to extract symptoms from free text
        selected = extract_symptoms_from_text(message)

    # If we found symptoms, use model to predict
    if selected:
        feat = {s: (1 if s in selected else 0) for s in SYMPTOMS}
        x_vec = vec.transform([feat])
        proba = model.predict_proba(x_vec)[0]
        preds = list(zip(model.classes_, proba))
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        top3 = [{"disease": p[0], "probability": round(float(p[1]), 4), "precautions": PRECAUTIONS.get(p[0], [])} for p in preds_sorted[:3]]

        reply = {
            "type": "diagnosis",
            "selected_symptoms": selected,
            "predictions": top3,
            "advice": "These are AI-based suggestions. Please consult a doctor for serious symptoms."
        }
        return jsonify(reply)

    # 2) No symptoms found — handle general queries
    low = message.lower()
    if any(w in low for w in ["hello", "hi", "hey"]):
        return jsonify({"type": "text", "reply": "Hello! How can I help you today? You can tell me your symptoms."})
    if any(w in low for w in ["thank", "thanks"]):
        return jsonify({"type": "text", "reply": "You're welcome! Do you have any other questions?"})
    if "what is" in low:
        return jsonify({"type": "text", "reply": "Please ask your question in simple words. If it’s medical, I can give suggestions based on symptoms."})

    # Fallback
    return jsonify({"type": "text", "reply": GENERIC_RESPONSES[0]})

if __name__ == "__main__":
    app.run(debug=True)
