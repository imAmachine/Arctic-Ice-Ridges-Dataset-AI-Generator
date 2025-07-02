import torch
import numpy
import torchvision.transforms as T
import torch.nn.functional as F

from abc import ABC, abstractmethod
from PIL import Image

from generativelib.model.arch.enums import ModelTypes
from generativelib.model.arch.common_transforms import get_infer_transforms
from generativelib.preprocessing.processors import *
from generativelib.preprocessing.preprocessor import DataPreprocessor

from src.config_deserializer import InferenceConfigDeserializer


class InferenceContext(ABC):
    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.generator = None
        self.image_size = 256
        self.outpainting_ratio = 0.0

    @staticmethod
    def load_model(model_name: str, config: InferenceConfigDeserializer, device: torch.device):
        from src.gan.gan_infer_context import GanInferenceContext
        from src.diffusion.diffusion_infer_context import DiffusionInferenceContext

        model_enum = ModelTypes[model_name.upper()]
        if model_enum == ModelTypes.GAN:
            return GanInferenceContext(config, device)
        elif model_enum == ModelTypes.DIFFUSION:
            return DiffusionInferenceContext(config, device)
        else:
            raise ValueError("Неизвестный тип модели")
        
    def _load_params(self):
        params = self.config.get_global_section("dataset")
        self.image_size = params.get("image_size", 256)
        self.outpainting_ratio = params.get("outpainting_size", 0.2)
        
    def _prepare_input_image(self, image: np.ndarray) -> torch.Tensor:
        infer_preprocessors = [
                RotateMask(),
                AdjustToContent(),
                Crop(k=1.0),
            ]
        
        infer_preprocessors.append(
            InferenceProcessor(outpaint_ratio=self.outpainting_ratio)
        )
        
        preprocessor = DataPreprocessor(processors=infer_preprocessors)
        preprocessed_img = preprocessor.process_image(image)
        transforms = get_infer_transforms(self.image_size)
        transform_img = transforms(preprocessed_img).to(self.device)
        mask = self._create_mask(transform_img)
        return transform_img, mask
    
    def _create_mask(self, inp: torch.Tensor) -> torch.Tensor:
        image = inp[0]
    
        h, w = image.shape
        mask = torch.ones_like(image)

        rows = (image != -1).any(dim=1).long() 
        cols = (image != -1).any(dim=0).long() 

        y_start = torch.argmax(rows)
        y_end = h - torch.argmax(rows.flip(0)) 
        x_start = torch.argmax(cols) 
        x_end = w - torch.argmax(cols.flip(0)) 

        # Зануляем центральный квадрат
        mask[y_start:y_end, x_start:x_end] = 0

        return mask.unsqueeze(0)
    
    def _postprocess(self, generated_tensor: torch.Tensor, original_mask) -> Image.Image:
        h, w = original_mask.shape
        resized = F.interpolate(generated_tensor, size=(int(h * (1 + self.outpainting_ratio)),int(w * (1 + self.outpainting_ratio))), mode='bilinear', align_corners=False)

        return T.ToPILImage()(resized.squeeze(0)).resize((self.image_size, self.image_size))
    
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def load_weights(self, path: str):
        pass

    @abstractmethod
    def generate_from_mask(self, image: numpy.ndarray) -> Image.Image:
        pass