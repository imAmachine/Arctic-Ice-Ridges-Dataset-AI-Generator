import os
from typing import Any, Callable, Dict, cast
import torch
from generativelib.config_tools.default_values import DATASET_KEY, PATH_KEY
from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.train.base import ModuleOptimizersCollection
from src.config_deserializer import TrainConfigDeserializer
from src.gan.gan_templates import GanTemplate
from generativelib.dataset.loader import DatasetCreator
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import CheckpointHook, TrainData, TrainManager, VisualizeHook
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *


# ВРЕМЕННОЕ (видимо постоянное) РЕШЕНИЕ, НУЖНО РАЗГРЕБАТЬ!!!!!!!!!!!!!!!
class GanTrainContext:
    def __init__(self, config_serializer: TrainConfigDeserializer):
        self.config_serializer = config_serializer

    def _preprocessor_metadata(self) -> Dict[str, Any]:
        paths = self.config_serializer.params_by_section(section='path', keys=['masks', 'dataset'])
        
        dataset_preprocessor = DataPreprocessor(
            *paths.values(),
            files_extensions=['.png'],
            processors=[
                RotateMask(),
                AdjustToContent(),
                Crop(k=0.5),
            ]
        )
        return dataset_preprocessor.get_metadata()

    def _model_template(self) -> GanTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.GAN)
        arch_collection = self.config_serializer.optimize_collection(ModelTypes.GAN)

        self._add_model_evaluators(arch_collection)

        return GanTemplate(model_params, arch_collection)

    def _add_model_evaluators(self, optimizers_collection: ModuleOptimizersCollection) -> None:
        """Добавляет лоссы для генератора и дискриминатора"""
        discriminator = optimizers_collection.by_type(GenerativeModules.GAN_DISCRIMINATOR).module
        
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
        device: torch.device,
        gen_func: Callable
    ) -> TrainData:
        # глобальные данные обучения и путей
        glob_train_params = self.config_serializer.get_global_section(ExecPhase.TRAIN.name.lower())
        glob_path_params = self.config_serializer.get_global_section(PATH_KEY)
        visualize_hook = self._visualize_hook(gen_func)
        checkpoint_hook = self._checkpoint_hook()
        
        # путь для вывода
        processed_path = glob_path_params.get('processed', '')
        model_output_path = ModelTypes.GAN.name.lower()
        final_output_path = os.path.join(processed_path, model_output_path)
        
        return TrainData(
            device=device,
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
        transforms = get_common_transforms(cast(int, img_size))
        
        template = self._model_template()
        train_data = self._train_data(device, template.gen_optim.module)
        
        ds_creator = self._dataset_creator(metadata, transforms)

        return TrainManager(
            optim_template=template, 
            train_data=train_data,
            dataloaders=ds_creator.create_loaders()
        )