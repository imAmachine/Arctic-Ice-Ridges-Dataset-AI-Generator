import tkinter as tk
import os
import cv2
import torch
import torchvision.transforms.v2 as tf2
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from tkinter import filedialog, messagebox
from torchvision.utils import save_image
from PIL import Image, ImageTk
from datetime import datetime
from config.preprocess import PREPROCESSORS

from src.common.enums import ModelType
from src.models.gan.architecture import CustomGenerator
from src.models.models import GAN
from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.processors import InferenceMaskingProcessor


class ImageGenerationApp:
    def __init__(self, root, config, model_weights_path):
        self.root = root
        self.root.title("Генератор изображений")
        self.config = config
        self.weights_loaded = model_weights_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.current_mask = None
        self.current_mask_path = None
        self.oupainting_size = 0.2
        self.image_size = 256
        self.last_generated_image = None
        
        self._setup_ui()

    def _setup_ui(self):
        self._create_load_buttons()
        self._create_display_labels()
        self._create_model_selector()
        self._create_action_buttons()
        self._create_log_box()

    def _create_load_buttons(self):
        load_frame = tk.Frame(self.root)
        load_frame.pack(pady=10)

        tk.Button(load_frame, text="Загрузка пользовательских весов", command=self.load_custom_weights).grid(row=0, column=0, padx=5)
        tk.Button(load_frame, text="Загрузить бинарную маску", command=self.load_mask).grid(row=0, column=1, padx=5)

    def _create_display_labels(self):
        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=10)

        tk.Label(display_frame, text="Входное изображение").grid(row=0, column=0)
        tk.Label(display_frame, text="Сгенерированное изображение").grid(row=0, column=1)

        self.mask_label = tk.Label(display_frame)
        self.mask_label.grid(row=1, column=0, padx=10)

        self.result_label = tk.Label(display_frame)
        self.result_label.grid(row=1, column=1, padx=10)

    def _create_model_selector(self):
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10)

        tk.Label(model_frame, text="Выберите модель:").grid(row=0, column=0)
        self.model_choice = tk.StringVar(value="Выберите модель")
        tk.OptionMenu(model_frame, self.model_choice, "GAN", "Diffusion", command=self.select_model).grid(row=0, column=1, padx=5)

    def _create_action_buttons(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        tk.Button(frame, text="Сгенерировать изображение", command=self.generate_image, height=2, width=25).grid(row=0, column=0, padx=10)
        tk.Button(frame, text="Сохранить изображение", command=self.save_image, height=2, width=25).grid(row=0, column=1, padx=10)

    def save_image(self):
        if self.last_generated_image is None:
            messagebox.showwarning("Внимание", "Сначала сгенерируйте изображение.")
            return
        mask_name = os.path.splitext(os.path.basename(self.current_mask_path))[0] + "_generated"

        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{mask_name}.png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if not path:
            return

        try:
            save_image(self.last_generated_image, path)
            self.log(f"💾 Изображение сохранено в: {path}")
        except Exception as e:
            self.log(f"❌ Ошибка при сохранении: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {str(e)}")

    def _create_log_box(self):
        self.log_box = tk.Text(self.root, height=8, width=80)
        self.log_box.pack(padx=10, pady=10)  

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_box.insert(tk.END, f"{timestamp} {message}\n")
        self.log_box.see(tk.END)

    def load_custom_weights(self):
        """Ручная загрузка пользовательских весов"""
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.model.checkpoint_load(path)
            self.weights_loaded = True
            self.log(f"✅ Custom weights loaded from: {path}")
        except Exception as e:
            self.log(f"❌ Error loading weights: {str(e)}")
            messagebox.showerror("Error", f"Failed to load weights: {str(e)}")

    def select_model(self, choice):
        if choice == "GAN":
            try:
                checkpoint_map = {
                    ModelType.GENERATOR: {
                        'model': ('trainers', ModelType.GENERATOR, 'module', 'arch'),
                    }
                }
                self.model = GAN(self.device, n_critic=5, checkpoint_map=checkpoint_map)
                self.model.build_train_modules(self.config['gan'])
                self.log("🧠 GAN-модель инициализирована.")
            except Exception as e:
                self.model = None
                self.log(f"❌ Ошибка инициализации GAN: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить GAN: {str(e)}")

        elif choice == "Diffusion":
            self.model = None
            self.log("ℹ️ Diffusion модель пока не реализована.")

        else:
            self.model = None
            self.log("⚠️ Модель не выбрана.")

    def load_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.tif"), ("All Files", "*.*")])
        if path:
            try:
                self.current_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                self.current_mask_path = path
                mask = Image.fromarray(self.current_mask).resize((self.image_size, self.image_size))
                photo = ImageTk.PhotoImage(mask)
                self._update_image_label(self.mask_label, mask)
                self.mask_label.config(image=photo)
                self.mask_label.image = photo
                self.log("📥 Маска успешно загружена")
            except Exception as e:
                self.log(f"❌ Ошибка загрузки маски.: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить маску: {str(e)}")

    def _update_image_label(self, label, image: Image.Image):
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

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
    
    def generate_image(self):
        if not self._check_generation_conditions():
            return
        try:
            img = self._prepare_input_image()
            with torch.no_grad():
                generated = self.model.trainers[ModelType.GENERATOR].module(img.unsqueeze(0))
            self._display_result(generated)
        except Exception as e:
            self.log(f"❌ Ошибка генерации: {str(e)}")
            messagebox.showerror("Ошибка", f"Генерация не удалась: {str(e)}")

    def _check_generation_conditions(self) -> bool:
        if not self.weights_loaded:
            messagebox.showerror("Ошибка", "Веса модели не загружены!")
            return False
        if self.current_mask is None:
            messagebox.showerror("Ошибка", "Загрузите маску!")
            return False
        if self.model is None:
            messagebox.showerror("Ошибка", "Выберете модель!")
            return False
        return True
    
    def _prepare_input_image(self) -> torch.Tensor:
        infer_preprocessors = PREPROCESSORS.copy()
        infer_preprocessors.append(InferenceMaskingProcessor(outpaint_ratio=self.oupainting_size))
        preprocessor = DataPreprocessor(processors=infer_preprocessors)
        preprocessed_img = preprocessor.process_image(self.current_mask)
        transforms = tf2.Compose(CustomGenerator.get_infer_transforms(self.image_size))
        transformed_img = transforms(preprocessed_img)
        return transformed_img.to(self.device)
    
    def _display_result(self, generated_img: torch.Tensor):
        original_shape = self.current_mask.shape
        resized = F.interpolate(generated_img, size=(int(original_shape[0] * (1 + self.oupainting_size)), int(original_shape[1] * (1 + self.oupainting_size))), mode='bilinear', align_corners=False)

        self.last_generated_image = resized 
        self.log(f"✅ Генерация завершена.")

        final_img = self.insert_orig_in_gen(resized)
        pil_image = T.ToPILImage()(final_img).resize((self.image_size, self.image_size))
        self._update_image_label(self.result_label, pil_image)