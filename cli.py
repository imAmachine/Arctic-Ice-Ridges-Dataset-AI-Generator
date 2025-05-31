import os
import argparse
from typing import Dict

from torch import cuda

from generativelib.config_tools.base import ConfigReader
from src.config_wrappers import TrainConfigSerializer
from generativelib.dataset.loader import DatasetCreator, DatasetMaskingProcessor
from generativelib.dataset.mask_processors import *
from generativelib.model.train.train import TrainConfigurator, TrainManager
from generativelib.preprocessing.preprocessor import DataPreprocessor

from generativelib.model.enums import ExecPhase
from generativelib.model.arch.enums import ModelTypes, GenerativeModules

from generativelib.preprocessing.processors import *
from src.tester import ParamGridTester
from src.model_templates import GANTrainTemplate

# Enable cuDNN autotuner for potential performance boost
torch.backends.cudnn.benchmark = True

DEVICE = 'cuda' if cuda.is_available() else 'cpu'
configs_folder_path = './configs'

train_cs = TrainConfigSerializer(configs_folder_path, ExecPhase.TRAIN)
test_config = ConfigReader(configs_folder_path, ExecPhase.TEST)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_weights', action='store_true')
    return parser.parse_args()

def init_dataset_metadata() -> Dict:
    paths = train_cs.params_by_section(section='path', keys=['masks', 'dataset'])
    
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

def get_model_train_template(model_type: str, img_size: int):
    train_template = None
    transforms = None
    
    if ModelTypes[model_type.upper()] == ModelTypes.GAN:
        transforms = GANTrainTemplate.get_transforms(img_size)
        arch_collection, model_params = train_cs.serialize_model(DEVICE, ModelTypes.GAN)
        train_template = GANTrainTemplate(model_params, arch_collection)
    
    return train_template, transforms

def main():
    args = parse_arguments()
    masking_processor = DatasetMaskingProcessor(processors=train_cs.serialize_mask_processors())
    train_configurator = TrainConfigurator(
                            device=DEVICE, 
                            **train_cs.get_global_section('train'),
                            **train_cs.params_by_section(section='path', keys=['vizualizations', 'weights'])
                        )
    
    dataset_metadata = init_dataset_metadata()
    dataset_params = train_cs.get_global_section("dataset")
    
    train_template, transforms = get_model_train_template(
        model_type=args.model, 
        **train_cs.params_by_section(section="arch", keys='img_size')
    )
    
    ds_creator = DatasetCreator(
        metadata=dataset_metadata,
        mask_processor=masking_processor,
        transforms=transforms,
        dataset_params=dataset_params
    ) 
    
    tr_man = TrainManager(
        train_template=train_template,
        train_configurator=train_configurator,
        dataloaders=ds_creator.create_loaders(),
    )
    
    tr_man.run()

if __name__ == '__main__':
    main()


# if args.test:
#     # Testing
#     print("Запуск тестов...")
#     tester = ParamGridTester(
#         param_grid_config=model_cfg,
#         trainer=trainer,
#         dataset=ds_creator,
#         output_folder_path=WEIGHTS_PATH,
#         seed=42,
#     )
#     tester.run()
    
# else:
# Training
# print(f"Обучение модели {args.model} на {args.epochs} эпохах...")
# train_template.build_train_modules(model_cfg)
# train_template._evaluators_from_config(model_cfg['evaluators_info'], device=DEVICE)

# if args.load_weights:
#     train_template.checkpoint_load(CHECKPOINT_PATH)
# trainer.run()

# if args.get_generator:
    #     checkpoint_map = {
    #         GenerativeModules.GENERATOR: {
    #             'model': ('trainers', GenerativeModules.GENERATOR, 'module', 'arch'),
    #         }
    #     }

    #     model = GAN(DEVICE, n_critic=5, checkpoint_map=checkpoint_map)
    #     model.build_train_modules(model_cfg)
        
    #     model.checkpoint_load(CHECKPOINT_PATH)
    #     model.checkpoint_save(os.path.join(WEIGHTS_PATH, 'generator.pt'))