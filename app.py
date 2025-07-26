from flask import Flask, request, jsonify, render_template
from flask import send_from_directory
from flask_cors import CORS
import replicate
import os
from predictor import predict_traits_from_wav
from pydub import AudioSegment
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

CORS(app)

replicate.Client(api_token="r8_dD6WHS74H4braB2xp9LHEJC8CYossoB0TuFKU")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['audio']
    raw_path="raw_audio.webm"
    wav_path = "recording.wav"
    # file.save(raw_path)
    # audio = AudioSegment.from_file(raw_path)
    # audio.export(wav_path, format="wav")
    # Call your model prediction function
    traits = predict_traits_from_wav(wav_path)

#     # Avatar generation
#     prompt = f"A {traits['energy'].lower()} energy, {traits['confidence'].lower()} confidence {traits['personality']} person, digital cartoon avatar, expressive face, clean background"
#     output = replicate.run(
#     "stability-ai/sdxl",
#     input={
#         "prompt": prompt,
#         "width": 512,
#         "height": 512,
#         "num_outputs": 1
#     }
# )
    # image_url = output[0] if output else ""

    return jsonify({
        "traits": traits

    })

if __name__ == "__main__":
    app.run(debug=True)

