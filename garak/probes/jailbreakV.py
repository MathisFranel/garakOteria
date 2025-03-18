from garak.probes.base import Probe
from datasets import load_dataset
from PIL import Image
import base64
import os

class JailBreakV(Probe):
    name = "jailbreakv_ollama"
    description = "Évaluation des MLLMs via Ollama avec JailBreakV-28K"
    bcp47 = "en"
    tags = ["multimodal", "jailbreak", "ollama"]

    def __init__(self):
        super().__init__()
        self.dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", split="train")
        self.image_dir = os.getenv("JailBreakV_IMAGE_DIR", "./jailbreakv_images/")

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def probe(self, model):
        for entry in self.dataset:
            image_path = os.path.join(self.image_dir, entry['image_path'])
            base64_image = self._encode_image(image_path) if entry['image_path'] else ""
            
            prompt = {
                "role": "user",
                "content": [
                    {"type": "text", "text": entry['jailbreak_query']},
                    {"type": "image", "source": {"type": "base64", "data": base64_image}}
                ]
            }
            
            yield self._craft_attempt(prompt, entry)

    def _craft_attempt(self, prompt, metadata):
        return Attempt(
            prompt=prompt,
            params={
                'attack_type': metadata['format'],
                'model': 'llava:34b-v1.6'  # Modèle Ollama cible
            }
        )