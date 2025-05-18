import os
import argparse

from torch import cuda
import torchvision.transforms.v2 as tf2

from config.default import DEFAULT_TEST_CONF, DEFAULT_TRAIN_CONF
from config.preprocess import MASKS_FILE_EXTENSIONS, PREPROCESSORS
from config.path import *

from src.common import Utils
from src.common.enums import ExecPhase
from src.dataset.loader import DatasetCreator, DatasetMaskingProcessor
from src.dataset.strategies import RandomHoleStrategy
from src.dataset.structs import ProcessingStrategies
from src.models.gan.architecture import CustomGenerator
from src.models.models import GAN
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
        model.load(checkpoint_path)
    else:
        print(f"Файл чекпоинта {checkpoint_path} не найден. Обучение с нуля.")

def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='GAN для сегментации ледовых торосов')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--train', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--augs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--val_rat', type=float, default=0.2)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--test', type=str)
    return parser.parse_args()

def main():
    args = parse_arguments()

    init_config()
    cfg = Utils.from_json(CONFIG)
    train_conf = cfg[ExecPhase.TRAIN.value]
    test_conf = cfg[ExecPhase.TEST.value]

    dataset_preprocessor = DataPreprocessor(
        MASKS_FOLDER_PATH, 
        PREPROCESSED_MASKS_FOLDER_PATH, 
        MASKS_FILE_EXTENSIONS, 
        PREPROCESSORS
    )
    
    if args.preprocess:
        print('Препроцессинг данных...')
        dataset_preprocessor.process_folder()

    # Training
    if args.train:
        config = train_conf[args.train]
        model = None
        masking_processor = None
        transforms = None
        
        print(f"Обучение модели {args.train} на {args.epochs} эпохах...")
        if args.train=='gan':
            model = GAN(DEVICE, n_critic=5)
            model.build(config)
            
            processing_strats = ProcessingStrategies([RandomHoleStrategy(strategy_name="holes")])
            masking_processor = DatasetMaskingProcessor(
                mask_params=config['mask_params'],
                processing_strats=processing_strats
            )
            transforms = tf2.Compose(CustomGenerator.get_transforms(config['target_image_size']))

        if args.load_weights:
            checkpoint_path = os.path.join(WEIGHTS_PATH, 'training_checkpoint.pt')
            load_checkpoint(model, checkpoint_path)
        
        ds_creator = DatasetCreator(
            input_preprocessor=dataset_preprocessor,
            masking_processor=masking_processor,
            model_transforms=transforms,
            augs_per_img=args.augs
        )
        
        trainer = Trainer(
            device=DEVICE,
            model=model,
            dataset=ds_creator,
            output_path=WEIGHTS_PATH,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_ratio=args.val_rat
        )
        trainer.run()
    
    if args.test:
        print("Запуск тестов...")
        
        ds_creator = DatasetCreator(
            input_preprocessor=dataset_preprocessor,
            masking_processor=masking_processor,
            model_transforms=transforms,
            augs_per_img=args.augs
        )
        
        trainer = Trainer(
            device=DEVICE,
            model=model,
            dataset=ds_creator,
            output_path=WEIGHTS_PATH,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_ratio=args.val_rat
        )
        
        config = test_conf[args.test]
        processing_strats = ProcessingStrategies([RandomHoleStrategy(strategy_name="holes")])
            
        masking_processor = DatasetMaskingProcessor(
            mask_params=test_conf['mask_params'],
            processing_strats=processing_strats
        )
        transforms = tf2.Compose(CustomGenerator.get_transforms(test_conf['target_image_size']))
        
        # tester = ParamGridTester(
        #     param_grid_config=config,
        #     trainer=trainer,
        #     output_folder_path=WEIGHTS_PATH,
        #     seed=42,
        # )
        # tester.run_grid_tests()

if __name__ == '__main__':
    main()
