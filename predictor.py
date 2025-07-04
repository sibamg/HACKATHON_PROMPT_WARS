import numpy as np
import librosa
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
def predict_traits_from_wav(audio_path):
    MODEL_PATH = "model.h5"

    ENCODERS_PATH = "label.pkl"
    AUDIO_PATH =audio_path
    model = load_model(MODEL_PATH)
    print("AUDIO PATH IS"+AUDIO_PATH)

    def extract_mfcc(file_path, max_pad_len=174):
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs

    with open("label.pkl", "rb") as f:
        le_emotion, le_intensity, le_statement, le_repetition, le_actor = pickle.load(f)

    test_features = extract_mfcc(AUDIO_PATH)
    test_features = test_features.reshape(1, 40, 174, 1)
    predictions = model.predict(test_features)
    emotion_idx = np.argmax(predictions[0])
    intensity_idx = np.argmax(predictions[1])
    statement_idx = np.argmax(predictions[2])
    repetition_idx = np.argmax(predictions[3])
    actor_idx = np.argmax(predictions[4])
    emotion_map = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }

    intensity_map = {1: "normal", 2: "strong"}
    statement_map = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}
    repetition_map = {1: "1st", 2: "2nd"}

    # Use predicted values
    emotion = emotion_map.get(emotion_idx + 1, "unknown")
    intensity = intensity_map.get(intensity_idx + 1, "unknown")
    statement = statement_map.get(statement_idx + 1, "unknown")
    repetition = repetition_map.get(repetition_idx + 1, "unknown")
    actor = actor_idx + 1

    # print("Emotion:", emotion)
    # print("Intensity:", intensity)
    # print("Statement:", statement)
    # print("Repetition:", repetition)
    actor ="Male" if actor % 2 == 1 else "Female"


    MODEL_PATH_2="model_anirudh.h5"
    with open("label_anirudh.pkl", "rb") as f:
        le_conf, le_energy, le_clarity, le_persona = pickle.load(f)

    model_anirudh=load_model(MODEL_PATH_2)
    def predict_persona_from_local_file(file_path, model, le_conf, le_energy, le_clarity, le_persona,emotion,intensity,statement,r,actor):
        try:
            # Step 1: Load and extract features
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            delta = librosa.feature.delta(mfcc)

            def pad(x): return np.pad(x, ((0, 0), (0, max(0, 174 - x.shape[1]))), mode='constant')[:, :174]
            features = np.vstack([pad(mfcc), pad(chroma), pad(contrast), pad(delta)])
            features = features[np.newaxis, ..., np.newaxis]  # (1, 99, 174, 1)

            # Step 2: Predict
            preds = model.predict(features)
            result = {
                "Confidence": le_conf.inverse_transform([np.argmax(preds[0])])[0],
                "Energy":     le_energy.inverse_transform([np.argmax(preds[1])])[0],
                "Clarity":    le_clarity.inverse_transform([np.argmax(preds[2])])[0],
                "Persona":    le_persona.inverse_transform([np.argmax(preds[3])])[0],
                "Emotion":emotion,
                "Intensity":intensity,
                "Statement":statement,
                "Repetition":r,
                "Gender":actor
            }
            return result

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    # ✅ Provide your .wav file path (update as needed)
    # Make sure this file exists

    # ✅ Run prediction
    result = predict_persona_from_local_file(
        file_path=AUDIO_PATH,
        model=model_anirudh,
        le_conf=le_conf,
        le_energy=le_energy,
        le_clarity=le_clarity,
        le_persona=le_persona,
        emotion=emotion,
        intensity=intensity,
        statement=statement,
        r=repetition,
        actor=actor

    )

    # ✅ Show result
    return result