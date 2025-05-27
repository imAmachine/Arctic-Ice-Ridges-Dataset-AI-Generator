import cv2
import torch
import torchvision.transforms.v2 as tf2
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from config.preprocess import PREPROCESSORS
from PIL import Image
from torchvision.utils import save_image

from src.common.enums import ModelType
from src.models.gan.architecture import CustomGenerator
from src.models.models import GAN
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.processors import InferenceProcessor

class InferenceManager:
    def __init__(self, config, image_size=256, outpainting_size=0.2):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.oupainting_size = outpainting_size

        self.model = None
        self.weights_loaded = None
        self.current_mask = None
        self.current_mask_path = None
        self.last_generated_image = None

    def load_model(self, model_type: str):
        if model_type == "GAN":
            checkpoint_map = {
                ModelType.GENERATOR: {
                    'model': ('trainers', ModelType.GENERATOR, 'module', 'arch'),
                }
            }
            self.model = GAN(self.device, n_critic=5, checkpoint_map=checkpoint_map)
            self.model.build_train_modules(self.config['gan'])
        elif model_type == "Diffusion":
            raise NotImplementedError("Diffusion модель пока не реализована.")
        else:
            raise ValueError("Неизвестный тип модели")
        
    def load_weights(self, path: str):
        if not self.model:
            raise RuntimeError("Сначала инициализируйте модель.")
        self.model.checkpoint_load(path)
        self.weights_loaded = True

    def load_mask(self, path: str) -> Image:
        self.current_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.current_mask_path = path
        mask = Image.fromarray(self.current_mask).resize((self.image_size, self.image_size))
        return mask
    
    def generate(self):
        img = self._prepare_input_image()
        with torch.no_grad():
                generated = self.model.trainers[ModelType.GENERATOR].module(img.unsqueeze(0))
        pil_image = self._display_result(generated)
        return pil_image
    
    def _prepare_input_image(self) -> torch.Tensor:
        infer_preprocessors = PREPROCESSORS.copy()
        infer_preprocessors.append(InferenceProcessor(outpaint_ratio=self.oupainting_size))
        preprocessor = DataPreprocessor(processors=infer_preprocessors)
        preprocessed_img = preprocessor.process_image(self.current_mask)
        transforms = tf2.Compose(CustomGenerator.get_infer_transforms(self.image_size))
        transformed_img = transforms(preprocessed_img)
        return transformed_img.to(self.device)
    
    def _display_result(self, generated_img: torch.Tensor):
        original_shape = self.current_mask.shape
        resized = F.interpolate(generated_img, size=(int(original_shape[0] * (1 + self.oupainting_size)), int(original_shape[1] * (1 + self.oupainting_size))), mode='bilinear', align_corners=False)

        self.last_generated_image = resized

        final_img = self.insert_orig_in_gen(resized)
        pil_image = T.ToPILImage()(final_img).resize((self.image_size, self.image_size))
        return pil_image

    def insert_orig_in_gen(self, image: torch.Tensor):
        gen_img = image.squeeze(0)
        orig_img = torch.tensor(self.current_mask, dtype=torch.float32) / 255.0
        orig_img = orig_img.unsqueeze(0)

        _, h_big, w_big = gen_img.shape
        _, h, w = orig_img.shape
        y_offset = (h_big - h) // 2
        x_offset = (w_big - w) // 2
        gen_img[:, y_offset:y_offset + h, x_offset:x_offset + w] = orig_img
        return gen_img
    
    def save_last_image(self, path: str):
        if self.last_generated_image is None:
            raise RuntimeError("Нет изображения для сохранения.")
        save_image(self.last_generated_image, path)