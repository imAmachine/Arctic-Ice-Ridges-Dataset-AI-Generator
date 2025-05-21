import os
import argparse

from torch import cuda
import torchvision.transforms.v2 as tf2

from config.default import DEFAULT_TEST_CONF, DEFAULT_TRAIN_CONF
from config.preprocess import MASKS_FILE_EXTENSIONS, PREPROCESSORS
from config.path import *

from src.common import Utils
from src.common.enums import ExecPhase, ModelType
from src.dataset.loader import DatasetCreator, DatasetMaskingProcessor, IceRidgeDataset
from src.dataset.strategies import *
from src.models.gan.architecture import CustomGenerator
from src.models.models import GAN
from src.models.tester import ParamGridTester
from src.models.train import Trainer
from src.preprocessing.preprocessor import DataPreprocessor

DEVICE = 'cuda' if cuda.is_available() else 'cpu'


def validate_or_reset_section(config: dict, section: str, defaults: dict) -> None:
    """Ensure config[section] matches defaults; prompt to reset if not."""
    sec = config.get(section, {})
    if not isinstance(sec, dict) or set(sec.keys()) != set(defaults.keys()):
        ans = input(f"Конфигурация '{section}' некорректна. Перезаписать стандартными значениями? (Y/N): ")
        if ans.strip().upper() == 'Y':
            config[section] = defaults

def init_config():
    default_test_conf = {
        model_name: {
            **DEFAULT_TRAIN_CONF[model_name],
            **DEFAULT_TEST_CONF[model_name]
        }
        for model_name in DEFAULT_TRAIN_CONF
    }

    if not os.path.exists(CONFIG):
        print('Создаем файл конфигурации по умолчанию')
        Utils.to_json({
            ExecPhase.TRAIN.value: DEFAULT_TRAIN_CONF,
            ExecPhase.TEST.value: default_test_conf
        }, CONFIG)
        return
    
    cfg = Utils.from_json(CONFIG)
    validate_or_reset_section(cfg, ExecPhase.TRAIN.value, DEFAULT_TRAIN_CONF)
    
    # Валидация секции TEST с учетом дефолтных значений
    validate_or_reset_section(cfg, ExecPhase.TEST.value, default_test_conf)
    Utils.to_json(cfg, CONFIG)
    
def load_checkpoint(model: GAN, checkpoint_path: str) -> None:
    """Load model weights from checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        print(f"Загрузка весов из {checkpoint_path}")
        model.checkpoint_load(checkpoint_path)
    else:
        print(f"Файл чекпоинта {checkpoint_path} не найден. Обучение с нуля.")

def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='GAN для сегментации ледовых торосов')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--augs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--val_rat', type=float, default=0.2)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--get_generator', action='store_true')
    return parser.parse_args()

def init_mask_processors(config):
    masking_processor = DatasetMaskingProcessor(
        processors=[
            # EllipsoidPadding(**config[Padding.__name__]),
            RandomHoles(**config[RandomHoles.__name__]),
        ]
    )
    
    return masking_processor

def main():
    args = parse_arguments()

    init_config()
    cfg = Utils.from_json(CONFIG)
    phase_cfg = cfg[ExecPhase.TEST.value if args.test else ExecPhase.TRAIN.value]
    model_cfg = phase_cfg[args.model]
    
    dataset_preprocessor = DataPreprocessor(
        MASKS_FOLDER_PATH, 
        PREPROCESSED_MASKS_FOLDER_PATH, 
        MASKS_FILE_EXTENSIONS, 
        PREPROCESSORS
    )
    dataset_metadata = dataset_preprocessor.get_metadata()
    
    if args.get_generator:
        checkpoint_path = os.path.join(WEIGHTS_PATH, 'training_checkpoint.pt')

        checkpoint_map = {
            ModelType.GENERATOR: {
                'model': ('trainers', ModelType.GENERATOR, 'module', 'arch'),
            }
        }

        model = GAN(DEVICE, n_critic=5, checkpoint_map=checkpoint_map)
        model.build_train_modules(model_cfg)

        load_checkpoint(model, checkpoint_path)
        model.checkpoint_save(os.path.join(WEIGHTS_PATH, 'generator.pt'))
    
    model = None
    masking_processor = None
    transforms = None
    
    if args.model == 'gan':
        model = GAN(DEVICE, n_critic=5)
        masking_processor = init_mask_processors(config=model_cfg["mask_processors"])
        transforms = tf2.Compose(CustomGenerator.get_transforms(model_cfg['target_image_size']))
    
        if args.load_weights:
            checkpoint_path = os.path.join(WEIGHTS_PATH, 'training_checkpoint.pt')
            load_checkpoint(model, checkpoint_path)
        
        ds_creator = DatasetCreator(
            metadata=dataset_metadata,
            mask_processor=masking_processor,
            transforms=transforms,
            augs_per_img=args.augs,
            valid_size_p=0.2,
            shuffle=True,
            batch_size=args.batch_size,
            workers=4,
        )
        
        trainer = Trainer(
            device=DEVICE,
            model=model,
            dataloaders=ds_creator.create_loaders(),
            output_path=WEIGHTS_PATH,
            epochs=args.epochs,
            checkpoints_ratio=model_cfg["checkpoints_ratio"]
        )
    
    if args.test:
        # Testing
        print("Запуск тестов...")
        tester = ParamGridTester(
            param_grid_config=model_cfg,
            trainer=trainer,
            dataset=ds_creator,
            output_folder_path=WEIGHTS_PATH,
            seed=42,
        )
        tester.run()
    else:
        # Training
        print(f"Обучение модели {args.model} на {args.epochs} эпохах...")
        model.build_train_modules(model_cfg)
        trainer.run()

if __name__ == '__main__':
    main()
