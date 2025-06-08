import torch
import torchvision.transforms as T
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Dict, List
from PIL import Image

from generativelib.config_tools.default_values import PATH_KEY, WEIGHT_KEY
from src.config_deserializer import InferenceConfigDeserializer
from generativelib.dataset.base import BaseMaskProcessor
from generativelib.dataset.loader import DatasetCreator
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.arch.common_transforms import get_infer_transforms
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import TrainConfigurator
from generativelib.preprocessing.processors import *
from generativelib.preprocessing.preprocessor import DataPreprocessor


class TrainContext(ABC):
    def __init__(self, config_serializer):
        super().__init__()
        self.config_serializer = config_serializer

    def _preprocessor_metadata(self) -> Dict:
        paths = self.config_serializer.params_by_section(section='path', keys=['masks', 'dataset'])
        dataset_preprocessor = DataPreprocessor(
            *paths.values(),
            files_extensions=['.png'],
            processors=[RotateMask(), AdjustToContent(), Crop(k=0.5)]
        )
        return dataset_preprocessor.get_metadata()
    
    def _dataset_creator(self, dataset_metadata: Dict, transforms) -> DatasetCreator:
        mask_processors: List[BaseMaskProcessor] = self.config_serializer.all_dataset_masks()
        dataset_params = self.config_serializer.get_global_section("dataset")
        
        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processors=mask_processors,
            transforms=transforms,
            dataset_params=dataset_params
        )
    
    def _train_configurator(self, device: torch.device):
        train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        
        return TrainConfigurator(
            device=device, 
            **train_params,
            weights=self.config_serializer.params_by_section(section=PATH_KEY, keys=WEIGHT_KEY)
        )


class InferenceContext(ABC):
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def load_model(model_name: str, config: InferenceConfigDeserializer):
        from src.gan.gan_context import GanInferenceContext

        model_enum = ModelTypes[model_name.upper()]
        if model_enum == ModelTypes.GAN:
            return GanInferenceContext(config)
        elif model_name == "Diffusion":
            raise NotImplementedError("Diffusion модель пока не реализована.")
        else:
            raise ValueError("Неизвестный тип модели")
        
    def _load_params(self):
        params = self.config.get_global_section("dataset")
        self.image_size = params.get("outpainting_size", 256)
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
    def load_weights(self, path: str):
        pass

    @abstractmethod
    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        pass