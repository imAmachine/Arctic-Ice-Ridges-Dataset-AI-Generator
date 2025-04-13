from src.datasets.dataset import DatasetCreator
from src.gan.model import GenerativeModel
from src.gan.train import GANTrainer
from settings import *
import argparse

def main():
   parser = argparse.ArgumentParser(description='GAN модель для генерации ледовых торосов')
   parser.add_argument('--preprocess', action='store_true', help='Препроцессинг исходных данных')
   parser.add_argument('--generate', action='store_true', help='Генерация аугментированных данных')
   parser.add_argument('--train', action='store_true', help='Обучение модели')
   parser.add_argument('--augmentations', type=int, default=10, help='Количество аугментаций на изображение')
   parser.add_argument('--epochs', type=int, default=20000, help='Количество эпох обучения')
   parser.add_argument('--batch_size', type=int, default=4, help='Размер батча')
   parser.add_argument('--load_weights', action='store_true', help='Загрузить сохраненные веса модели')
   parser.add_argument('--device', type=str, default=DEVICE, help='Устройство для обучения (cuda/cpu)')
   
   args = parser.parse_args()
   
   run_all = not (args.preprocess or args.generate or args.train)
   
   model_gan = GenerativeModel(target_image_size=448, 
                             g_feature_maps=48, 
                             d_feature_maps=48,
                             device=args.device)
   
   ds_creator = DatasetCreator(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                       original_data_path=MASKS_FOLDER_PATH,
                       preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                       images_extentions=MASKS_FILE_EXTENSIONS,
                       model_transforms=model_gan.get_transforms(),
                       preprocessors=PREPROCESSORS,
                       augmentations_pipeline=AUGMENTATIONS,
                       device=args.device)
   
   if args.preprocess or run_all:
       print("Выполняется препроцессинг данных...")
       ds_creator.preprocess_data()
   
   if args.generate or run_all:
       print(f"Генерация аугментированных данных ({args.augmentations} на изображение)...")
       ds_creator.augmentate_data(augmentations_per_image=args.augmentations)
   
   if args.train or run_all:
       print(f"Запуск обучения модели на {args.epochs} эпох...")
       trainer = GANTrainer(model=model_gan, 
                            dataset_processor=ds_creator,
                            output_path=WEIGHTS_PATH,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            load_weights=args.load_weights)
       
       trainer.train()

if __name__ == "__main__":
   main()