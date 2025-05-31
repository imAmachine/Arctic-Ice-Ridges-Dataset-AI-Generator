from typing import Dict
import torch
from src.config_wrappers import TrainConfigSerializer
from generativelib.dataset.loader import DatasetCreator, DatasetMaskingProcessor
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import TrainConfigurator, TrainManager
from generativelib.preprocessing.preprocessor import DataPreprocessor
from generativelib.preprocessing.processors import *
from src.model_templates import GANTrainTemplate


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
    
    def init_train(self, model_type: str,):
        # предобработка и подгрузка метаданных
        metadata = self._preprocessor_metadata()
        
        template, transforms = self._model_template(
            model_type, 
            self.config_serializer.params_by_section(section="arch", keys='img_size'))
        
        # создание менеджера датасета
        ds_creator = self._dataset_creator(metadata, transforms)
        
        return self._train_manager(template, ds_creator.create_loaders())