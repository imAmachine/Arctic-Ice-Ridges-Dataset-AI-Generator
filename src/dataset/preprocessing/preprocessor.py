import os
from typing import Any, Dict, List
from cv2 import imwrite, IMREAD_GRAYSCALE
import numpy as np

from src.dataset.preprocessing.processors import IProcessor
from src.common.utils import Utils


class IceRidgeDatasetPreprocessor:    
    def __init__(self, processors: List[IProcessor]=None):
        self.metadata = {}
        if processors is None:
            self.processors: List[IProcessor] = []
        else:
            self.processors: List[IProcessor] = processors
    
    def add_processor(self, processor: IProcessor) -> None:
        self.processors.append(processor)
    
    def add_processors(self, processors: List[IProcessor]) -> None:
        self.processors.extend(processors)
    
    def process_image(self, image: np.ndarray, filename: str = '', output_path: str = '') -> np.ndarray:
        processing_img = image
        current_metadata = {}
        
        for processor in self.processors:
            processing_img = processor.process(processing_img, current_metadata)
        
        current_metadata.update({'path': output_path})
        self.metadata[filename] = current_metadata
        return processing_img
    
    def _get_output_path(self, filename: str, output_folder_path: str):
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_processed{ext}"
        return os.path.join(output_folder_path, new_filename)
    
    def _write_processed_img(self, image: np.ndarray, output_path: str):
        try:
            imwrite(output_path, image)
        except Exception as e:
            print(f"Error processing {output_path}: {str(e)}")
    
    def _process_file(self, input_path: str, output_folder: str, filename: str):
            print(f'Обработка {filename}')
            processing_img = Utils.cv2_load_image(os.path.join(input_path, filename), cv2_read_mode=IMREAD_GRAYSCALE)
            output_path = self._get_output_path(filename=filename, output_folder_path=output_folder)
            processing_img = self.process_image(processing_img, filename, output_path)
            self._write_processed_img(processing_img, output_path)
            
    
    def process_folder(self, input_folder: str, output_folder: str, files_extensions: List[str] = None) -> Dict[str, Dict[str, Any]]:
        if files_extensions is None:
            raise ValueError('Input files extensions not defined!')
        
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in os.listdir(input_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in files_extensions:
                self._process_file(input_folder, output_folder, filename)
        
        return self.metadata