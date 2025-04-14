from src.gan.dataset import DatasetCreator
from src.gan.model import GenerativeModel, GenerativeModelInference
from src.gan.train import GANTrainer
from settings import *
import argparse
import os
import torch
from torchvision.utils import save_image


def main():
    parser = argparse.ArgumentParser(description='GAN модель для генерации ледовых торосов')
    parser.add_argument('--preprocess', action='store_true', help='Препроцессинг исходных данных')
    parser.add_argument('--train', action='store_true', help='Обучение модели')
    parser.add_argument('--infer', action='store_true', help='Инференс сгенерированных изображений')
    parser.add_argument('--epochs', type=int, default=20000, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=1, help='Размер батча')
    parser.add_argument('--load_weights', action='store_true', help='Загрузить сохраненные веса модели')
    parser.add_argument('--input_image', type=str, help='Путь к изображению для инференса')
    parser.add_argument('--output_image', type=str, help='Путь для сохранения сгенерированного изображения')

    args = parser.parse_args()

    run_all = not (args.preprocess or args.train or args.infer)

    # Инициализация модели
    model_gan = GenerativeModel(target_image_size=256, 
                                g_feature_maps=64, 
                                d_feature_maps=32,
                                device=DEVICE,
                                lr=0.0005,
                                n_critic=5,
                                lambda_w=2.0,
                                lambda_bce=3.0,
                                lambda_gp=5.0,
                                lambda_l1=1.5)

    # Инициализация создателя датасета
    ds_creator = DatasetCreator(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                                 original_data_path=MASKS_FOLDER_PATH,
                                 preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                                 images_extentions=MASKS_FILE_EXTENSIONS,
                                 model_transforms=model_gan.get_transforms(),
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

    # Инференс
    if args.infer:
        print("Запуск инференса...")
        
        # Загружаем обученную модель
        model_gan.load_checkpoint(WEIGHTS_PATH)
        model_transforms = model_gan.get_transforms()
        inference_model = GenerativeModelInference(model_gan, device=DEVICE)
    
        # Инициализация DatasetCreator
        ds_creator = DatasetCreator(
            generated_path='./generated_images',
            original_data_path=MASKS_FOLDER_PATH,
            preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
            images_extentions=['.png'],
            model_transforms=model_transforms,
            preprocessors=PREPROCESSORS,
            augmentations=None,
            device=DEVICE,
        )

        # Получение dataloaders
        loader, _ = ds_creator.create_dataloaders(
            batch_size=1,
            shuffle=False,
            workers=1,
            random_select=False,
            val_ratio=0,
            val_augmentations=False,
            train_augmentations=False
        )
        
        with torch.no_grad():
            for i, (damaged, original, mask) in enumerate(loader):
                damaged = damaged.to(DEVICE)
                original = original.to(DEVICE)
                mask = mask.to(DEVICE)
                
                # Выполнение инференса
                generated_image = inference_model.generate_image(damaged, mask)
        
                save_image(generated_image, f'{args.output_image}_{i}.png')
                print(f"Сгенерированное изображение сохранено по пути: {args.output_image}_{i}.png")

if __name__ == "__main__":
    main()
