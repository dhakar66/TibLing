from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os

app = Flask(__name__)

# Initialize the model pipeline
try:
    # Use GPU if available, otherwise fallback to CPU
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("automatic-speech-recognition", model="openpecha/wav2vec2_run9", device=device)
except Exception as e:
    app.logger.error(f"Error initializing the model pipeline: {str(e)}")
    pipe = None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not pipe:
        return jsonify({"error": "Model pipeline not initialized"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
#

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the audio file with the model
        audio_file = file.read()
        results = pipe(audio_file)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": "Failed to process file"}), 500

if __name__ == '__main__':
    app.run(debug=True)
