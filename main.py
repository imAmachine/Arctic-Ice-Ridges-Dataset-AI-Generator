import os
import argparse
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf2
from src.models.train import GAN, Trainer
from src.models.gan_arch import WGanCritic, WGanGenerator
from src.models.structs import ArchModule
from src.models.custom_evaluators import *
from settings import *

from src.common.utils import Utils
from src.common.enums import ExecPhase as phases, ModelType
from src.dataset.dataset import DatasetCreator


def validate_or_reset_section(config: dict, section: str, defaults: dict) -> None:
    """Ensure config[section] matches defaults; prompt to reset if not."""
    sec = config.get(section, {})
    if not isinstance(sec, dict) or set(sec.keys()) != set(defaults.keys()):
        ans = input(f"Конфигурация '{section}' некорректна. Перезаписать стандартными значениями? (Y/N): ")
        if ans.strip().upper() == 'Y':
            config[section] = defaults


def init_config():
    if not os.path.exists(CONFIG):
        print('Создаем файл конфигурации по умолчанию')
        Utils.to_json({phases.TRAIN.value: DEFAULT_TRAIN_CONF,
                       phases.TEST.value: DEFAULT_TEST_CONF}, CONFIG)
        return
    cfg = Utils.from_json(CONFIG)
    for section, defaults in [(phases.TRAIN.value, DEFAULT_TRAIN_CONF),
                              (phases.TEST.value, DEFAULT_TEST_CONF)]:
        validate_or_reset_section(cfg, section, defaults)
    Utils.to_json(cfg, CONFIG)


def create_optimizer(parameters, lr: float=0.0001, betas=(0.0, 0.9)):
    """Create Adam optimizer with specified parameters."""
    return torch.optim.Adam(parameters, lr=lr, betas=betas)


def create_scheduler(optimizer, mode: str, factor: float=0.5, patience: int=6):
    """Create ReduceLROnPlateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience
    )


def build_evaluators(discriminator: WGanCritic) -> Dict:
    """Create dictionary of evaluation metrics and losses."""
    return {
        LossName.ADVERSARIAL.value: AdversarialLoss(discriminator),
        LossName.BCE.value: nn.BCELoss(),
        LossName.L1.value: nn.L1Loss(),
        LossName.WASSERSTEIN.value: WassersteinLoss(discriminator),
        LossName.GP.value: GradientPenalty(discriminator),
        MetricName.PRECISION.value: sklearn_wrapper(precision_score, DEVICE),
        MetricName.F1.value: sklearn_wrapper(f1_score, DEVICE),
        MetricName.IOU.value: sklearn_wrapper(jaccard_score, DEVICE),
    }


def build_gan_modules(config_section: dict) -> List[ArchModule]:
    """Construct ArchModule list for GAN model."""
    # Model initialization
    gen = WGanGenerator(
        input_channels=1,
        feature_maps=config_section['model_base_features']
    ).to(DEVICE)
    disc = WGanCritic(
        input_channels=1,
        feature_maps=config_section['model_base_features']
    ).to(DEVICE)
    
    g_optimizer = create_optimizer(gen.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
    g_scheduler = create_scheduler(g_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

    d_optimizer = create_optimizer(disc.parameters(), config_section['optimization_params']['lr'], betas=(0.0, 0.9))
    d_scheduler = create_scheduler(d_optimizer, config_section['optimization_params']['mode'], factor=0.5, patience=6)

    evaluators = build_evaluators(disc)

    modules: List[ArchModule] = []
    modules.extend([
        ArchModule(
            model_type=ModelType.GENERATOR,
            arch=gen,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            eval_funcs=evaluators,
            eval_settings=config_section['evaluators_info'][ModelType.GENERATOR.value]
        ),
        ArchModule(
            model_type=ModelType.DISCRIMINATOR,
            arch=disc,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            eval_funcs=evaluators,
            eval_settings=config_section['evaluators_info'][ModelType.DISCRIMINATOR.value]
        )
    ])
    
    return modules


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
    parser.add_argument('--train', type=str, default='gan')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--augs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--val_rat', type=float, default=0.2)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()

    init_config()
    cfg = Utils.from_json(CONFIG)
    train_conf = cfg[phases.TRAIN.value][args.train]

    # Transforms
    transforms = tf2.Compose(
        WGanGenerator.get_train_transforms(
            train_conf['target_image_size']
        )
    )

    # Dataset
    ds = DatasetCreator(
        device=DEVICE,
        output_path=AUGMENTED_DATASET_FOLDER_PATH,
        original_images_path=MASKS_FOLDER_PATH,
        preprocessed_images_path=PREPROCESSED_MASKS_FOLDER_PATH,
        images_ext=MASKS_FILE_EXTENSIONS,
        model_transforms=transforms,
        preprocessors=PREPROCESSORS,
        augs_per_img=args.augs
    )
    if args.preprocess:
        print('Препроцессинг данных...')
        ds.preprocess_data()

    # Training
    if args.train:
        print(f"Обучение модели {args.train} на {args.epochs} эпохах...")
        if args.train=='gan':
            modules = build_gan_modules(train_conf)
            model = GAN(DEVICE, modules, n_critic=5)

        if args.load_weights:
            checkpoint_path = os.path.join(WEIGHTS_PATH, 'training_checkpoint.pt')
            load_checkpoint(model, checkpoint_path)
        
        trainer = Trainer(
            device=DEVICE,
            model=model,
            dataset=ds,
            output_path=WEIGHTS_PATH,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_ratio=args.val_rat
        )
        trainer.run()

if __name__ == '__main__':
    main()
