import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any
from PIL import Image

from generativelib.model.arch.enums import ModelTypes
from generativelib.model.arch.common_transforms import get_infer_transforms
from generativelib.preprocessing.processors import *
from generativelib.preprocessing.preprocessor import DataPreprocessor

from src.config_deserializer import InferenceConfigDeserializer


class InferenceContext(ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.generator = None
        self.image_size = None
        self.outpainting_ratio = None

    @staticmethod
    def load_model(model_name: str, config: InferenceConfigDeserializer):
        from src.gan.gan_infer_context import GanInferenceContext
        from src.diffusion.diffusion_context import DiffusionInferenceContext

        model_enum = ModelTypes[model_name.upper()]
        if model_enum == ModelTypes.GAN:
            return GanInferenceContext(config)
        elif model_enum == ModelTypes.DIFFUSION:
            return DiffusionInferenceContext(config)
        else:
            raise ValueError("Неизвестный тип модели")
        
    def _load_params(self):
        params = self.config.get_global_section("dataset")
        self.image_size = params.get("image_size", 256)
        self.outpainting_ratio = params.get("outpainting_size", 0.2)
        
    def _prepare_input_image(self, image: torch.Tensor) -> torch.Tensor:
        infer_preprocessors = [
                RotateMask(),
                AdjustToContent(),
                Crop(k=0.5),
            ]
        infer_preprocessors.append(InferenceProcessor(outpaint_ratio=self.outpainting_ratio))
        preprocessor = DataPreprocessor(processors=infer_preprocessors)
        preprocessed_img = preprocessor.process_image(image)

        transforms = get_infer_transforms(self.image_size)
        return transforms(preprocessed_img).to(self.device)
        
    def _insert_original_mask(self, gen_img: torch.Tensor, orig_np) -> torch.Tensor:
        gen = gen_img.squeeze(0)
        orig = torch.tensor(orig_np, dtype=torch.float32) / 255.0
        orig = orig.unsqueeze(0)

        _, H_big, W_big = gen.shape
        _, H, W = orig.shape
        y_offset = (H_big - H) // 2
        x_offset = (W_big - W) // 2
        gen[:, y_offset:y_offset + H, x_offset:x_offset + W] = orig
        return gen
    
    def _postprocess(self, generated_tensor: torch.Tensor, original_mask) -> Image.Image:
        h, w = original_mask.shape
        resized = F.interpolate(generated_tensor, size=(int(h * (1 + self.outpainting_ratio)),int(w * (1 + self.outpainting_ratio))), mode='bilinear', align_corners=False)

        final = self._insert_original_mask(resized, original_mask)
        return T.ToPILImage()(final).resize((self.image_size, self.image_size))
    
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def load_weights(self, path: str):
        pass

    @abstractmethod
    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        pass