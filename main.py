from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.gan.dataset import DatasetCreator, InferenceMaskingProcessor
from src.gan.model import GenerativeModel
from src.gan.tester import ParamGridTester
from src.gan.train import GANTrainer
from src.gan.arch import AUGMENTATIONS
from src.common.utils import Utils
from src.common.structs import ExecPhase as phases

import argparse
import tkinter as tk

from gui.gui import ImageGenerationApp
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
    parser.add_argument('--train', action='store_true', help='Обучение модели')
    # parser.add_argument('--infer', action='store_true', help='Инференс на одном изображении')
    parser.add_argument('--input_path', type=str, help='Путь к изображению для инференса')
    parser.add_argument('--epochs', type=int, default=1000, help='Количество эпох обучения')
    parser.add_argument('--augs', type=int, default=1, help='Количество аугментированных сэмплов на снимок, определяет итоговый размер датасета')
    parser.add_argument('--batch_size', type=int, default=3, help='Размер батча')
    parser.add_argument('--val_rat', type=float, default=0.2, help='Размер валидационной выборки в процентах')
    parser.add_argument('--load_weights', action='store_true', help='Загрузить сохраненные веса модели')
    # parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--test', action='store_true', help='Запуск тестов параметров')

    args = parser.parse_args()

    run_all = not (args.preprocess or args.train or args.infer or args.gui or args.test)

    init_config()
    config = dict(Utils.from_json(CONFIG))
    
    # Инициализация модели
    model_gan = GenerativeModel(**config[phases.TRAIN.value], device=DEVICE)

    # Инициализация создателя датасета
    ds_creator = DatasetCreator(generated_path=AUGMENTED_DATASET_FOLDER_PATH,
                                original_data_path=MASKS_FOLDER_PATH,
                                preprocessed_data_path=PREPROCESSED_MASKS_FOLDER_PATH,
                                images_extentions=MASKS_FILE_EXTENSIONS,
                                model_transforms=model_gan.generator.get_model_transforms(config.get(phases.TRAIN.value)
                                                                                          .get("target_image_size")),
                                preprocessors=PREPROCESSORS,
                                augmentations=AUGMENTATIONS,
                                augs_per_img=args.augs,
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
                             val_ratio=args.val_rat,
                             checkpoints_ratio=50)
        
        trainer.train()

    # # Инференс
    # if args.infer:
    #     if not args.input_path:
    #         print("Ошибка: для инференса необходимо указать --input_path путь к изображению.")
    #         return
    #     img = Utils.cv2_load_image(args.input_path, cv2.IMREAD_GRAYSCALE)
    #     preprocessor = IceRidgeDatasetPreprocessor(PREPROCESSORS)
    #     preprocessed_img = preprocessor.process_image(img)
        
    #     processor = InferenceMaskingProcessor(outpaint_ratio=0.2)
    #     generated, _ = model_gan.infer_generate(preprocessed_img=preprocessed_img, 
    #                                                    checkpoint_path=WEIGHTS_PATH, 
    #                                                    processor=processor)

    #     output_path = './data/inference/output.png'
    #     cv2.imwrite(output_path, generated)
    #     print(f"Генерация завершена. Результат сохранён в {output_path}")

    if args.gui:
        root = tk.Tk()
        _ = ImageGenerationApp(root, WEIGHTS_PATH, model_gan, args)
        root.mainloop()
        return
    
    if args.test:
        grid_params = config.get(phases.TEST.value)
        if grid_params:
            print('запуск тестирования')
            tester = ParamGridTester(grid_params)
            tester.run_grid_tests()
        else:
            print('проблема чтения параметров тестирования')
        return
        

if __name__ == "__main__":
    main()
