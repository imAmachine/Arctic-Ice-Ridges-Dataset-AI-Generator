import os
from PIL import Image
from typing import Any, Callable, Dict, cast
import torch
from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from src.config_deserializer import TrainConfigDeserializer
from src.gan.gan_templates import GanTemplate
from generativelib.dataset.loader import DatasetCreator
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainData, TrainManager, VisualizeHook
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *
from generativelib.model.inference.base import ModuleInference

from src.create_context import TrainContext, InferenceContext
from src.config_deserializer import TrainConfigDeserializer, InferenceConfigDeserializer
from src.gan.gan_templates import GanTemplate


class GanInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer):
        super().__init__(config)

        self._load_params()
        self._load_model()

    def _load_model(self):
        arch_module = self.config.create_arch_module(
            model_type=ModelTypes.GAN,
            module_name="gan_generator"
        )
        self.generator = ModuleInference(GenerativeModules.GAN_GENERATOR, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        tensor = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self.generator.generate(tensor.unsqueeze(0))
        return self._postprocess(generated, image)


# ВРЕМЕННОЕ (видимо постоянное) РЕШЕНИЕ, НУЖНО РАЗГРЕБАТЬ!!!!!!!!!!!!!!!
class GanTrainContext(TrainContext):
    def __init__(self, config_serializer: TrainConfigDeserializer):
        super().__init__(config_serializer)

    def _model_template(self) -> GanTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.GAN)
        arch_collection = self.config_serializer.optimize_collection(ModelTypes.GAN)
        
        template = GanTemplate(model_params, arch_collection)
        
        return template

    def _add_model_evaluators(self, template: GanTemplate) -> None:
        """Добавляет лоссы для генератора и дискриминатора"""
        discriminator = template.get_discr_optimizer().module
        optimizers_collection = template.model_optimizers
        
        optimizers_collection.add_evals({
            GenerativeModules.GAN_GENERATOR: [
                EvalItem(
                    GeneratorLoss(discriminator),
                    name=LossName.ADVERSARIAL.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                )
            ],
            GenerativeModules.GAN_DISCRIMINATOR: [
                EvalItem(
                    WassersteinLoss(discriminator),
                    name=LossName.WASSERSTEIN.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                ),
                EvalItem(
                    GradientPenalty(discriminator),
                    name=LossName.GRADIENT_PENALTY.name,
                    type=EvaluatorType.LOSS,
                    weight=10.0,
                    exec_phase=ExecPhase.TRAIN
                )
            ]
        })

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
    
    def _train_data(
        self,
        template: GanTemplate
    ) -> TrainData:
        # глобальные данные обучения и путей
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        glob_path_params = self.config_serializer.get_global_section(PATH_KEY)
        
        visualize_hook = self._visualize_hook(
            gen_callable=template.get_gen_optimizer().module
        )
        
        checkpoint_hook = self._checkpoint_hook()
        
        # путь для вывода
        processed_path = glob_path_params.get('processed', '')
        model_output_path = ModelTypes.GAN.name.lower()
        final_output_path = os.path.join(processed_path, model_output_path)
        
        return TrainData(
            epochs=glob_train_params.get('epochs', 1000),
            model_out_folder=final_output_path,
            visualize_hook=visualize_hook,
            checkpoint_hook=checkpoint_hook
        )

    def init_train(self, device: torch.device) -> TrainManager:
        metadata = self._preprocessor_metadata()
        
        img_size = self.config_serializer.params_by_section(
            section=DATASET_KEY, 
            keys='img_size'
        )
        transforms = get_common_transforms(cast(int, img_size)) # определение Compose трансформаций для GAN
        template = self._model_template() # создание шаблона обучения для GAN
        self._add_model_evaluators(template) # добавление специальных лоссов, свойственных архитектуре GAN
        
        train_data = self._train_data(template) # определение данных, необходимых в TrainManager
        ds_creator = self._dataset_creator(metadata, transforms) # создание менеджера датасета

        return TrainManager(
            device=device,
            optim_template=template, 
            train_data=train_data,
            dataloaders=ds_creator.create_loaders()
        )