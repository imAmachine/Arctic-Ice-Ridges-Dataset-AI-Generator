from src.gan.dataset import DatasetCreator
from src.gan.model import GenerativeModel
from src.gan.train import GANTrainer
from settings import *
import argparse
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser(description='GAN модель для генерации ледовых торосов')
    parser.add_argument('--preprocess', action='store_true', help='Препроцессинг исходных данных')
    parser.add_argument('--train', action='store_true', help='Обучение модели')
    parser.add_argument('--epochs', type=int, default=20000, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=1, help='Размер батча')
    parser.add_argument('--load_weights', action='store_true', help='Загрузить сохраненные веса модели')

    args = parser.parse_args()

    run_all = not (args.preprocess or args.train or args.infer)

    # Инициализация модели
    model_gan = GenerativeModel(target_image_size=256, 
                                g_feature_maps=64, 
                                d_feature_maps=32,
                                device=DEVICE,
                                lr=0.0005,
                                n_critic=5,
                                lambda_w=1.0,
                                lambda_bce=2.0,
                                lambda_gp=5.0,
                                lambda_l1=1.5)

    # Инициализация создателя датасета
    ds_creator = DatasetCreator(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                                 original_data_path=MASKS_FOLDER_PATH,
                                 preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                                 images_extentions=MASKS_FILE_EXTENSIONS,
                                 model_transforms=model_gan.model_transforms(),
                                 preprocessors=PREPROCESSORS,
                                 augmentations=AUGMENTATIONS,
                                 device=DEVICE)

    # Препроцессинг данных
    if args.preprocess or run_all:
        print("Выполняется препроцессинг данных...")
        ds_creator.preprocess_data()

    # Обучение модели
    if args.train or run_all:
        print(f"Запуск обучения модели на {args.epochs} эпох...")
        trainer = GANTrainer(model=model_gan, 
                             dataset_processor=ds_creator,
                             output_path=WEIGHTS_PATH,
                             epochs=args.epochs,
                             device=DEVICE,
                             batch_size=args.batch_size,
                             load_weights=args.load_weights,
                             val_ratio=0.1,
                             checkpoints_ratio=15)
        
        trainer.train()

if __name__ == "__main__":
    main()
