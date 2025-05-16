from typing import List, Literal
import torchvision
import torchvision.transforms.v2 as tf2
from src.gan.dataset import DatasetCreator
from src.gan.custom_evaluators import *
from src.gan.arch import WGanCritic, WGanGenerator
from src.gan.model import ArchModule, GANModel
from src.gan.train import ModelTrainer
from src.common.utils import Utils
from src.common.structs import ExecPhase as phases

import argparse
from settings import *

def validate_or_reset_config_section(config: dict, section_name: str, default_conf: dict) -> None:
    section = config.get(section_name)
    expected_keys = set(default_conf.keys())
    actual_keys = set(section.keys()) if isinstance(section, dict) else set()

    if actual_keys != expected_keys:
        answer = input(f"Конфигурация '{section_name}' некорректна. Перезаписать стандартными значениями? (Y/N): ")
        if answer.strip().upper() == "Y":
            config[section_name] = default_conf


def init_config():
    if not os.path.exists(CONFIG):
        print('Файл конфигурации отсутствует, будет создан стандартный')
        Utils.to_json({phases.TRAIN.value: DEFAULT_TRAIN_CONF, phases.TEST.value: DEFAULT_TEST_CONF}, CONFIG)
        return

    config = Utils.from_json(CONFIG)

    validate_or_reset_config_section(config, phases.TRAIN.value, DEFAULT_TRAIN_CONF)
    validate_or_reset_config_section(config, phases.TEST.value, DEFAULT_TEST_CONF)

    Utils.to_json(config, CONFIG)
            

def main():
    parser = argparse.ArgumentParser(description='GAN модель для генерации ледовых торосов')
    parser.add_argument('--preprocess', action='store_true', help='Препроцессинг исходных данных')
    parser.add_argument('--train', type=str, default='gan', help='Обучение выбранной модели')
    parser.add_argument('--input_path', type=str, help='Путь к изображению для инференса')
    parser.add_argument('--epochs', type=int, default=1000, help='Количество эпох обучения')
    parser.add_argument('--augs', type=int, default=1, help='Количество аугментированных сэмплов на снимок, определяет итоговый размер датасета')
    parser.add_argument('--batch_size', type=int, default=3, help='Размер батча')
    parser.add_argument('--val_rat', type=float, default=0.2, help='Размер валидационной выборки в процентах')
    parser.add_argument('--load_weights', action='store_true', help='Загрузить сохраненные веса модели')
    parser.add_argument('--test', action='store_true', help='Запуск тестов параметров')

    args = parser.parse_args()

    run_all = not (args.preprocess or args.train or args.infer or args.gui or args.test)

    init_config()
    config = dict(Utils.from_json(CONFIG))
    model_transforms = tf2.Compose(WGanGenerator.get_train_transforms(config[phases.TRAIN.value][args.train].get("target_image_size")))
    
    # Инициализация создателя датасета
    ds_creator = DatasetCreator(
        device=DEVICE,
        output_path=AUGMENTED_DATASET_FOLDER_PATH,
        original_images_path=MASKS_FOLDER_PATH,
        preprocessed_images_path=PREPROCESSED_MASKS_FOLDER_PATH,
        images_ext=MASKS_FILE_EXTENSIONS,
        model_transforms=model_transforms,
        preprocessors=PREPROCESSORS,
        augs_per_img=args.augs,
    )

    # Препроцессинг данных
    if args.preprocess or run_all:
        print("Выполняется препроцессинг данных...")
        ds_creator.preprocess_data()

    # Обучение модели
    if args.train or run_all:
        model = None
        config = config[phases.TRAIN.value][args.train]
        
        def get_module(model_type, arch, optimizer, scheduler, evaluators, evaluators_config):
            return ArchModule(
                model_type, 
                arch,
                optimizer,
                scheduler,
                evaluators,
                evaluators_config
            )
        
        if args.train == 'gan':            
            arch_list = {
                ModelType.GENERATOR: WGanGenerator(input_channels=1, feature_maps=config.get('model_base_features')).to(DEVICE),
                ModelType.DISCRIMINATOR: WGanCritic(input_channels=1, feature_maps=config.get('model_base_features')).to(DEVICE)
            }
            
            optimizer = torch.optim.Adam(arch_list[ModelType.GENERATOR].parameters(), lr=config.get('optimization_params').get('lr'), betas=(0.0, 0.9))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.get('optimization_params').get('mode'), factor=0.5, patience=6)
            
            evaluators = {
                LossName.ADVERSARIAL.value: AdversarialLoss(arch_list[ModelType.DISCRIMINATOR]),
                LossName.BCE.value: nn.BCELoss(),
                LossName.L1.value: nn.L1Loss(),
                LossName.WASSERSTEIN.value: WassersteinLoss(arch_list[ModelType.DISCRIMINATOR]),
                LossName.GP.value: GradientPenalty(arch_list[ModelType.DISCRIMINATOR]),
                MetricName.PRECISION.value: sklearn_wrapper(precision_score, DEVICE),
                MetricName.F1.value: sklearn_wrapper(f1_score, DEVICE),
                MetricName.IOU.value: sklearn_wrapper(jaccard_score, DEVICE),
            }
            
            modules: List[ArchModule] = []
            
            for t, m in arch_list.items():
                evaluators_config = config['evaluators_info'][t.value]
                modules.append(get_module(
                    t,
                    m,
                    optimizer,
                    scheduler,
                    evaluators,
                    evaluators_config
                ))
            
            
            model = GANModel(device=DEVICE, modules=modules, output_path=WEIGHTS_PATH, n_critic=5)
            
        print(f"Запуск обучения модели на {args.epochs} эпох...")
        trainer = ModelTrainer(
            device=DEVICE,
            generative_model=model,
            dataset_processor=ds_creator,
            output_path=WEIGHTS_PATH,
            epochs=args.epochs,
            batch_size=args.batch_size,
            load_weights=args.load_weights,
            validation_ratio=args.val_rat,
            checkpoints_freq=50
        )
        trainer.train()
        

if __name__ == "__main__":
    main()
