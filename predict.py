from cog import BasePredictor, Input

import sys
import os
import torch
import torchaudio

sys.path.extend([
    "/src/tortoise-tts/tortoise",
])

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.cache/'

from api import TextToSpeech
from utils.audio import load_voices


MODELS_DIR = "/src/.models"


class Predictor(BasePredictor):
    
    def setup(self):
        self.tts = TextToSpeech(models_dir=MODELS_DIR)

    def predict(self, text: str = Input(description="Text to prefix with 'hello '")) -> str:

        import time

        t1 = time.time()

        text = f"Hello, I am a large language model named {text}"
        voice = "random"
        preset = "fast"
        output_path = "/src/results/"
        model_dir = "/src/models"
        candidates = 3
        seed = None
        produce_debug_state = True
        cvvp_amount = 0.0
        
        selected_voices = voice.split(',')

        k = 0
        selected_voice = selected_voices[0]
        
        #for k, selected_voice in enumerate(selected_voices):
        print("VOICE ", k)

        t2 = time.time()

        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        
        voice_samples, conditioning_latents = load_voices(voice_sel)

        t3 = time.time()

        gen, dbg_state, _ = self.tts.tts_with_preset(
            text, 
            k=candidates, 
            voice_samples=voice_samples, 
            conditioning_latents=conditioning_latents,
            preset=preset, 
            use_deterministic_seed=seed, 
            return_deterministic_state=True, 
            cvvp_amount=cvvp_amount
        )
        
        t4 = time.time()

        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)
        
        t5 = time.time()

        if produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

        t6 = time.time()

        print(f"t2-t1: {t2-t1}")
        print(f"t3-t2: {t3-t2}")
        print(f"t4-t3: {t4-t3}")
        print(f"t5-t4: {t5-t4}")
        print(f"t6-t5: {t6-t5}")

        return os.path.join(output_path, f'{selected_voice}_{k}.wav')
