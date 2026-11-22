from typing import Optional

from .deepfake_extractor import DeepfakeDetector

AI_IMAGE_MODEL = "Ateeqq/ai-vs-human-image-detector"


class AiImageDetector(DeepfakeDetector):
    def __init__(
        self,
        model: str = AI_IMAGE_MODEL,
        device: str = "cpu",
        models_dir: Optional[str] = None,
        fake_label: Optional[str] = None,
    ):
        """
        Obrazy wygenerowane przez AI (Stable Diffusion, Midjourney, DALL-E), nie podmienione twarze.
        Gotowy klasyfikator z Hugging Face, ten sam mechanizm i katalog modeli co deepfake.
        Podaj `fake_label`, gdy model nazywa klasy tylko LABEL_0/LABEL_1.
        Wynik to wskaźnik: detektory słabo uogólniają na generatory spoza zbioru uczącego.
        """
        super().__init__(image_model=model, device=device, models_dir=models_dir, image_fake_label=fake_label)
