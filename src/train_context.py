from typing import Dict
import torch
import os

from src.config_wrappers import TrainConfigSerializer
from generativelib.dataset.loader import DatasetCreator, DatasetMaskingProcessor
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import TrainConfigurator, TrainManager
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *
from src.model_templates import GANTrainTemplate, DiffusionTrainTemplate


class AppTrainContext:
    def __init__(self, config_serializer: TrainConfigSerializer):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config_serializer = config_serializer
    
    def _preprocessor_metadata(self) -> Dict:
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
    
    def _model_template(self, model_type: str, img_size: int) -> tuple:
        if ModelTypes[model_type.upper()] == ModelTypes.GAN:
            transforms = GANTrainTemplate.get_transforms(img_size)
            arch_collection, model_params = self.config_serializer.model_serialize(self.device, ModelTypes.GAN)
            train_template = GANTrainTemplate(model_params, arch_collection)
        elif ModelTypes[model_type.upper()] == ModelTypes.DIFFUSION:
            transforms = DiffusionTrainTemplate.get_transforms(img_size)
            arch_collection, model_params = self.config_serializer.model_serialize(self.device, ModelTypes.DIFFUSION)
            train_template = DiffusionTrainTemplate(model_params, arch_collection)
        
        return train_template, transforms
    
    def _dataset_creator(self, dataset_metadata: Dict, transforms) -> DatasetCreator:
        dataset_params = self.config_serializer.get_global_section("dataset")
        masking_processor = DatasetMaskingProcessor(self.config_serializer._serialize_mask_processors())
        
        return DatasetCreator(
            metadata=dataset_metadata,
            mask_processor=masking_processor,
            transforms=transforms,
            dataset_params=dataset_params
        )
    
    def _train_manager(self, train_template, dataloaders: Dict[ExecPhase, Dict]) -> TrainManager:
        train_configurator = TrainConfigurator(
            device=self.device, 
            **self.config_serializer.get_global_section('train'),
            **self.config_serializer.params_by_section(section='path', keys=['vizualizations', 'weights'])
        )
        
        return TrainManager(
            train_template=train_template,
            train_configurator=train_configurator,
            dataloaders=dataloaders,
        )
    
    def init_train(self, model_params: Dict):
        # предобработка и подгрузка метаданных
        metadata = self._preprocessor_metadata()
        
        # получение текущего шаблона для обучения
        template, transforms = self._model_template(
            model_params.model, 
            self.config_serializer.params_by_section(section="arch", keys='img_size')
        )

        if model_params.load_weights:
            template.arch_optimizers.checkpoint_load(os.path.join(self.config_serializer.params_by_section(section='path', keys='weights'), 'training_checkpoint.pt'), device=self.device)
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        
        return self._train_manager(
            template, 
            ds_creator.create_loaders()
        )