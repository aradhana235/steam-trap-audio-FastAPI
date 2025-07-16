from fastapi import FastAPI
from pydantic import BaseModel
import requests
import joblib
import os
from api.classify import extract_features
import traceback
import subprocess

app = FastAPI()

class AudioRequest(BaseModel):
    audio_file_url: str
    device_id: str
    tb_token: str

model = joblib.load("model/audio_model.pkl")

@app.post("/classify")
async def classify(request: AudioRequest):
    audio_url = request.audio_file_url
    device_id = request.device_id
    tb_token = request.tb_token

    mp3_filename = f"{device_id}.mp3"
    wav_filename = f"{device_id}.wav"

    try:
        # Step 1: Download audio
        r = requests.get(audio_url)
        with open(mp3_filename, "wb") as f:
            f.write(r.content)

        # Step 2: Convert MP3 to WAV using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_filename, wav_filename
        ], check=True)

        # Step 3: Extract features and predict
        features = extract_features(wav_filename)
        status = model.predict([features])[0]

        # Cleanup
        os.remove(mp3_filename)
        os.remove(wav_filename)

        # Step 4: Send to ThingsBoard
        payload = {
            "trap_status": status,
            "status_color": {
                "normal": "🟢",
                "leak": "🟡",
                "blocked": "🔴"
            }.get(status, "⚪")
        }

        tb_url = f"https://eu.thingsboard.cloud/api/v1/{tb_token}/telemetry"
        res = requests.post(tb_url, json=payload)
        print("✅ Sent to ThingsBoard:", res.status_code)

        return {"trap_status": status}

    except Exception as e:
        traceback.print_exc()  # Show full error in terminal
        return {"error": str(e)}  # Return error message to Swagger
