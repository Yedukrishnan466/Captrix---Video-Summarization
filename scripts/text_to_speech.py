from gtts import gTTS
import os
from pathlib import Path

def say(text, filename, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    audio_file = base_name + ".mp3"
    output_path = out_dir / audio_file

    tts = gTTS(
        text=text,
        lang="en",
        slow=False
    )

    tts.save(str(output_path))

    print(f"Audio saved at: {output_path}")
