from gtts import gTTS
import io
import time
from deep_translator import GoogleTranslator
from backend.evaluation.log_metrics import log_metric
from backend.experiments.experiment_tracker import log_experiment

def text_to_speech(text, lang="en"):
    start = time.time()
    latency = 0.0
    audio_bytes = None
    try:
        
        if lang != "en":
            text_to_speak = GoogleTranslator(source='en', target=lang).translate(text)
            print(f"Translated '{text}' -> '{text_to_speak}' ({lang})")
        else:
            text_to_speak = text

        tts = gTTS(text=text_to_speak, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        
    except Exception as e:
        print("TTS/Translation ERROR:", str(e))
        return None
    finally:
        
        latency = round(time.time() - start, 2)
        try:
            log_metric(
                agent="tts",
                model_version="audio_v1",
                confidence=0.95,
                object_count=None,
                latency=latency,
                hallucination=False
            )
            log_experiment(
                agent="tts",
                confidence=0.95,
                latency=latency,
                hallucination=False,
                model_version="audio_v1"
            )
        except Exception as log_err:
                print("Metric logging failed:", log_err)

    return audio_bytes