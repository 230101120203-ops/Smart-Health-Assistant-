# train_model.py
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "body_pain",
    "rash", "sneezing", "nausea", "vomiting", "joint_pain", "sore_throat", "diarrhea"
]

DISEASES = [
    "Common Cold",
    "Flu",
    "Dengue",
    "Malaria",
    "Allergy",
    "Gastroenteritis",
    "Strep Throat"
]

def generate_sample():
    sample = {s: 0 for s in SYMPTOMS}
    disease = random.choice(DISEASES)
    if disease == "Common Cold":
        for s in ["cough", "sneezing", "sore_throat"]:
            if random.random() < 0.8: sample[s] = 1
        if random.random() < 0.3: sample["fever"] = 1
    elif disease == "Flu":
        for s in ["fever", "cough", "fatigue", "body_pain", "headache"]:
            if random.random() < 0.8: sample[s] = 1
    elif disease == "Dengue":
        for s in ["fever", "headache", "body_pain", "rash", "joint_pain", "nausea"]:
            if random.random() < 0.85: sample[s] = 1
    elif disease == "Malaria":
        for s in ["fever", "headache", "fatigue", "nausea", "vomiting"]:
            if random.random() < 0.75: sample[s] = 1
    elif disease == "Allergy":
        for s in ["sneezing", "cough", "headache", "rash"]:
            if random.random() < 0.7: sample[s] = 1
    elif disease == "Gastroenteritis":
        for s in ["nausea", "vomiting", "diarrhea", "fatigue"]:
            if random.random() < 0.85: sample[s] = 1
    elif disease == "Strep Throat":
        for s in ["sore_throat", "fever", "headache"]:
            if random.random() < 0.8: sample[s] = 1

    # small noise
    for s in SYMPTOMS:
        if random.random() < 0.02:
            sample[s] = 1 - sample[s]
    return sample, disease

def build_dataset(n=4000):
    X, y = [], []
    for _ in range(n):
        s, d = generate_sample()
        X.append(s)
        y.append(d)
    return pd.DataFrame(X), pd.Series(y)

if __name__ == "__main__":
    print("Building dataset...")
    X, y = build_dataset(4000)
    print(y.value_counts())
    vec = DictVectorizer(sparse=False)
    X_vec = vec.fit_transform(X.to_dict(orient="records"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = "model.joblib"
    vec_path = "vectorizer.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(vec, vec_path)
    print("Saved:", os.path.abspath(model_path), os.path.abspath(vec_path))
