import os
import json
import pickle
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

# Load the model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Directory containing audio files
directory = 'talks'
transcriptions = {}

for filename in os.listdir(directory):
    if filename.endswith(".mp3"):
        print(f"Processing {filename}")
        # Load and preprocess the audio file
        audio_file = os.path.join(directory, filename)
        audio, sr = librosa.load(audio_file, sr=16000)  # Resample to 16kHz

        # Split audio into 30-second chunks
        chunk_duration = 30  # seconds
        chunk_size = chunk_duration * sr  # Number of samples per chunk
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

        # Process each chunk and generate transcription
        chunk_transcriptions = []
        for i, chunk in enumerate(chunks):
            print(i)
            input_features = processor.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                generated_ids = model.generate(input_features, return_timestamps=True, language='en')
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=False)
            chunk_transcriptions.append(" ".join(transcription))  # Combine parts if necessary

        # Join all chunk transcriptions back into a single string
        full_transcription = " ".join(chunk_transcriptions)
        transcriptions[filename] = full_transcription
    else:
        continue

# Save the transcriptions to a JSON file
with open("transcripts.json", "w") as f:
    json.dump(transcriptions, f, indent=4)

# Save the transcriptions to a pickle file
with open("transcriptions.pkl", "wb") as f:
    pickle.dump(transcriptions, f)