import sys
import os

sys.path.append("/src/tortoise-tts/tortoise")
os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.cache/'

from cog import BasePredictor, Input, Path
import tempfile
import torch
import torchaudio
from api import TextToSpeech
from utils.audio import load_voices

MODELS_DIR = "/src/.models"


import dataclasses

@dataclasses.dataclass
class Output:
    output: Path
    voice: str


class Predictor(BasePredictor):
    
    def setup(self):
        self.tts = TextToSpeech(models_dir=MODELS_DIR)

    def predict(
        self, 
        text: str = Input(description="Input text"),
        voice: str = Input(description="Voice to use", default="random", choices=["random"]),
        preset: str = Input(description="Preset to use", default="fast", choices=["ultra_fast", "fast", "standard", "high_quality"]),
        seed: int = Input(description="Seed for deterministic generation", default=None),
        candidates: int = Input(description="Number of candidates to generate", default=1),
    ) -> Output:

        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / f'{voice}.wav'

        voice_samples, conditioning_latents = load_voices([voice])

        gen = self.tts.tts_with_preset(
            text, 
            k=candidates, 
            voice_samples=voice_samples, 
            conditioning_latents=conditioning_latents,
            preset=preset, 
            use_deterministic_seed=seed, 
            return_deterministic_state=True, 
            cvvp_amount=0.0
        )

        torchaudio.save(out_path, gen.squeeze(0).cpu(), 24000)
        
        return Output(
            output=out_path
            voice=voice
        )
