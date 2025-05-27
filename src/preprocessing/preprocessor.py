import os
from typing import Any, Dict, List
from cv2 import imwrite, IMREAD_GRAYSCALE
import numpy as np

from src.common.utils import Utils
from src.preprocessing.base import Processor


class DataPreprocessor:
    def __init__(self, input_folder_path: str = None, output_folder_path: str = '', files_extensions: List[str] = None, processors: List[Processor] = None):
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.input_path = input_folder_path
        self.output_path = output_folder_path
        self.metadata_json_path = os.path.join(self.output_path, 'metadata.json')
        
        self.files_extensions = files_extensions
        self.processors: List[Processor] = processors if processors else []

    def add_processor(self, processor: Processor) -> None:
        self.processors.append(processor)

    def add_processors(self, processors: List[Processor]) -> None:
        self.processors.extend(processors)

    def process_image(self, image: np.ndarray, filename: str = "", file_output_path: str = "") -> np.ndarray:
        metadata = {}
        for processor in self.processors:
            for dep in getattr(processor, "dependencies", []):
                if dep.__name__ not in metadata:
                    raise RuntimeError(f"Processor '{processor.name}' requires '{dep.__name__}' to be applied first.")

            image = processor.process(image)
            metadata[processor.name] = processor.get_metadata_value()

        metadata["path"] = file_output_path
        if filename:
            self.metadata[filename] = metadata
        return image

    def _get_output_path(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        return os.path.join(self.output_path, f"{base}_processed{ext}")

    def _write_processed_img(self, image: np.ndarray, file_output_path: str) -> None:
        try:
            imwrite(file_output_path, image)
        except Exception as e:
            print(f"[ERROR] Failed to write {file_output_path}: {str(e)}")

    def _process_file(self, filename: str) -> None:
        print(f"[INFO] Обработка {filename}")
        image = Utils.cv2_load_image(os.path.join(self.input_path, filename), IMREAD_GRAYSCALE)
        file_output_path = self._get_output_path(filename)
        processed_image = self.process_image(image, filename, file_output_path)
        self._write_processed_img(processed_image, file_output_path)

    def is_metadata_exist(self) -> bool:
        metadata = None

        if os.path.exists(self.metadata_json_path):
            metadata = Utils.from_json(self.metadata_json_path)
        
        if metadata is None:
            return False
        
        if len(metadata.items()) <= 0:
            return False
        
        return True

    def process_folder(self) -> None:
        if not self.files_extensions:
            raise ValueError("Расширения файлов не представлены!")

        if self.processors is None or len(self.processors) <= 0:
            raise ValueError('Пайплайн препроцессоров не объявлен!')
        
        os.makedirs(self.output_path, exist_ok=True)
        
        print('Препроцессинг данных...')
        for filename in os.listdir(self.input_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.files_extensions:
                self._process_file(filename)

        if len(self.metadata) == 0:
            raise ValueError(f'[{DataPreprocessor.__class__.__name__}] Метаданные после предобработки пусты. Остановка.')
        
        Utils.to_json(self.metadata, self.metadata_json_path)
    
    def get_metadata(self):
        if self.is_metadata_exist():
            self.metadata = Utils.from_json(self.metadata_json_path)
        else:
            self.process_folder()
        
        return self.metadata