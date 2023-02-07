import sys
import os

sys.path.append("/src/tortoise-tts/tortoise")
os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.cache/'

from cog import BasePredictor, BaseModel, File, Input, Path

import requests
import tempfile
import torch
import torchaudio
from api import TextToSpeech
from utils.audio import load_voices, load_audio

MODELS_DIR = "/src/.models"


def download_audio(voice_url, voice_dir):
    filename = voice_url.split('/')[-1]+'.wav'
    filepath = voice_dir / filename
    if filepath.exists():
        return filepath
    audio_file = requests.get(voice_url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(audio_file.read())
    return filepath


class Predictor(BasePredictor):
    
    def setup(self):
        self.tts = TextToSpeech(models_dir=MODELS_DIR)

    def predict(
        self, 
        text: str = Input(description="Input text"),
        voice: str = Input(description="Voice to use", default="random", choices=["random", "clone"]),
        voice_file_urls: str = Input(description="Voice clone files", default=None),
        preset: str = Input(description="Preset to use", default="standard", choices=["ultra_fast", "fast", "standard", "high_quality"]),
        seed: int = Input(description="Seed for deterministic generation", default=None),
        #candidates: int = Input(description="Number of candidates to generate", default=1),
    ) -> Path:

        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / f'{voice}.wav'

        voice_samples = []
        conditioning_latents = None

        if voice == 'random':
            voice_samples, conditioning_latents = load_voices([voice])

        elif voice == 'clone':
            voice_dir = Path('voice_files')
            voice_dir.mkdir(exist_ok=True)
            voice_file_urls = voice_file_urls.split('|')

            for voice_url in voice_file_urls:
                voice_file = download_audio(voice_url, voice_dir)
                sample = load_audio(str(voice_file), 22050)
                voice_samples.append(sample)

        gen = self.tts.tts_with_preset(
            text, 
            k=1, 
            voice_samples=voice_samples, 
            conditioning_latents=conditioning_latents,
            preset=preset, 
            use_deterministic_seed=seed, 
            return_deterministic_state=True, 
            cvvp_amount=0.0
        )

        torchaudio.save(
            out_path, 
            gen.squeeze(0).cpu(), 
            24000
        )
        print(f"Saved to {out_path}")

        return out_path
        