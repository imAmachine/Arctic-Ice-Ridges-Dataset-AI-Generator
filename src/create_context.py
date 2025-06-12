import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Dict, Callable, Any
from PIL import Image

from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.dataset.loader import DatasetCreator
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.arch.common_transforms import get_infer_transforms
from generativelib.model.enums import ExecPhase
from generativelib.model.train.base import OptimizationTemplate
from generativelib.model.train.train import CheckpointHook, TrainData, TrainManager, VisualizeHook
from generativelib.preprocessing.processors import *
from generativelib.preprocessing.preprocessor import DataPreprocessor

from src.config_deserializer import InferenceConfigDeserializer, TrainConfigDeserializer


class TrainContext(ABC):
    def __init__(self, config_serializer: TrainConfigDeserializer, model_type: ModelTypes):
        super().__init__()
        self.config_serializer = config_serializer
        self.model_type = model_type

    def _preprocessor_metadata(self) -> Dict[str, Any]:
        paths = self.config_serializer.params_by_section(section='path', keys=['masks', 'dataset'])
        dataset_preprocessor = DataPreprocessor(
            *paths.values(),
            files_extensions=['.png'],
            processors=[RotateMask(), AdjustToContent(), Crop(k=0.5)]
        )
        return dataset_preprocessor.get_metadata()
    
    def _dataset_creator(self, dataset_metadata: Dict[str, Any], transforms) -> DatasetCreator:
        mask_processors = self.config_serializer.all_dataset_masks()
        dataset_params = self.config_serializer.get_global_section("dataset")

        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processors=mask_processors,
            transforms=transforms,
            dataset_params=dataset_params
        )

    def _visualize_hook(
        self,
        gen_callable: Callable
    ) -> VisualizeHook:
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        visualize_interval = glob_train_params.get('visualize_interval', 5)
        
        return VisualizeHook(
            generate_fn=gen_callable,
            interval=visualize_interval
        )

    def _checkpoint_hook(self) -> CheckpointHook:
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        checkpoint_interval = glob_train_params.get('checkpoint_interval', 25)
        return CheckpointHook(checkpoint_interval)
    
    def init_train(self, device: torch.device) -> TrainManager:
        metadata = self._preprocessor_metadata()
        
        img_size = self.config_serializer.params_by_section(
            section=DATASET_KEY, 
            keys='img_size'
        )
        transforms = self._get_model_transform(img_size) # определение Compose трансформаций для модели
        template = self._model_template() # создание шаблона обучения для модели
        self._add_model_evaluators(template) # добавление специальных лоссов, свойственных архитектуре модели
        
        train_data = self._train_data(template) # определение данных, необходимых в TrainManager
        ds_creator = self._dataset_creator(metadata, transforms) # создание менеджера датасета

        return TrainManager(
            device=device,
            optim_template=template, 
            train_data=train_data,
            dataloaders=ds_creator.create_loaders()
        )
    
    def _train_data(
        self,
        template: OptimizationTemplate
    ) -> TrainData:
        # глобальные данные обучения и путей
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        glob_path_params = self.config_serializer.get_global_section(PATH_KEY)

        visualize_hook = self._visualize_for_model(template)
        
        checkpoint_hook = self._checkpoint_hook()
        
        # путь для вывода
        processed_path = glob_path_params.get('processed', '')
        model_output_path = self.model_type.name.lower()
        final_output_path = os.path.join(processed_path, model_output_path)
        
        return TrainData(
            epochs=glob_train_params.get('epochs', 1000),
            model_out_folder=final_output_path,
            visualize_hook=visualize_hook,
            checkpoint_hook=checkpoint_hook
        )
    
    @abstractmethod
    def _add_model_evaluators(self, template: OptimizationTemplate) -> None:
        pass

    @abstractmethod
    def _get_model_transform(self, img_size: int) -> T.Compose:
        pass

    @abstractmethod
    def _model_template(self) -> OptimizationTemplate:
        pass

    @abstractmethod
    def _visualize_for_model(self, template: OptimizationTemplate) -> VisualizeHook:
        pass


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
        from src.gan.gan_context import GanInferenceContext
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