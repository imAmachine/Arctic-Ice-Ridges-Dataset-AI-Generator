import cv2

from PIL import Image

from src.create_context import InferenceContext
from src.config_deserializer import InferenceConfigDeserializer


class InferenceManager:
    def __init__(self, config: InferenceConfigDeserializer):
        self.config = config

        self.model = None
        self.weights_loaded = False
        self.current_mask = None
        self.current_mask_path = None
        self.last_generated_image = None

    def load_model(self, model_name: str):
        self.model = InferenceContext.load_model(
            model_name=model_name,
            config=self.config
        )

    def load_weights(self, path: str):
        if not self.model:
            raise RuntimeError("Модель не инициализирована.")
        self.model.load_weights(path)
        self.weights_loaded = True
    
    def load_mask(self, path: str) -> Image.Image:
        self.current_mask_path = path
        self.current_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(self.current_mask)
    
    def generate(self) -> Image.Image:
        if not self.model:
            raise RuntimeError("Модель не инициализирована.")
        if self.current_mask is None:
            raise RuntimeError("Маска не загружена.")
        self.last_generated_image = self.model.generate_from_mask(self.current_mask)
        return self.last_generated_image

    def save_last_image(self, path: str):
        if self.last_generated_image is None:
            raise RuntimeError("Нет изображения для сохранения.")
        self.last_generated_image.save(path)