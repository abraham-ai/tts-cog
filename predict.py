import sys
import os

sys.path.append("/src/tortoise-tts/tortoise")
os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.cache/'

MODELS_DIR = "/src/.models"

from cog import BasePredictor, BaseModel, File, Input, Path

import requests
import tempfile
import torch
import torchaudio
from typing import Iterator, Optional
from api import TextToSpeech
from utils.audio import load_voices, load_audio


def download(url, folder, ext):
    filename = url.split('/')[-1]+ext
    filepath = folder / filename
    if filepath.exists():
        return filepath
    raw_file = requests.get(url, stream=True).raw
    with open(filepath, 'wb') as f:
        f.write(raw_file.read())
    return filepath


class CogOutput(BaseModel):
    file: Path
    name: Optional[str] = None
    thumbnail: Optional[Path] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False


class Predictor(BasePredictor):
    
    preset_voices = os.listdir('preset_voices')
    voice_options = ['random', 'clone', *preset_voices]

    def setup(self):
        print("cog:setup")
        self.tts = TextToSpeech(models_dir=MODELS_DIR)
        print("TTS loaded")

    def predict(
        self, 
        text: str = Input(description="Input text"),
        voice: str = Input(description="Voice to use", default="random", choices=voice_options),
        voice_file_urls: str = Input(description="Voice clone files", default=None),
        preset: str = Input(description="Preset to use", default="standard", choices=["ultra_fast", "fast", "standard", "high_quality"]),
        seed: int = Input(description="Seed for deterministic generation", default=None),
        #candidates: int = Input(description="Number of candidates to generate", default=1),
    ) -> Iterator[CogOutput]:

        print("Running TTS...")

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
                voice_file = download(voice_url, voice_dir, '.wav')
                sample = load_audio(str(voice_file), 22050)
                voice_samples.append(sample)

        else:
            voice_dir = os.path.join('preset_voices', voice)
            for voice_file in os.listdir(voice_dir):
                voice_file = os.path.join(voice_dir, voice_file)
                sample = load_audio(str(voice_file), 22050)
                voice_samples.append(sample)
            print(f"Got {len(voice_samples)} voice samples")

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

        yield CogOutput(file=out_path, thumbnail=out_path, name=text, attributes=None, progress=1.0, isFinal=True)
        