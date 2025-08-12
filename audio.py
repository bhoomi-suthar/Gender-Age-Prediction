import sounddevice as sd
import numpy as np
import librosa

DURATION = 4  # seconds
SAMPLE_RATE = 22050  # Hz

def record_audio():
    print("Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete!")
    return np.squeeze(audio)

def analyze_gender(audio_data):
    try:
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=SAMPLE_RATE)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        
        if len(pitch_values) == 0:
            return "Unknown (no pitch detected)"
        
        avg_pitch = np.mean(pitch_values)
        print(f"Average Pitch: {avg_pitch:.2f} Hz")

        if avg_pitch < 165:
            return "Male"
        elif avg_pitch > 180:
            return "Female"
        else:
            return "Uncertain"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    audio_clip = record_audio()
    gender_prediction = analyze_gender(audio_clip)
    print(f"Predicted Gender: {gender_prediction}")
