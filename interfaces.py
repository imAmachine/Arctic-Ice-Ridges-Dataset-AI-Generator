import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import torch
from torchvision.transforms import ToPILImage
from datetime import datetime

from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.gan.dataset import InferenceMaskingProcessor
from settings import PREPROCESSORS

from src.preprocessing.preprocessor import IceRidgeDatasetPreprocessor
from src.gan.dataset import InferenceMaskingProcessor
from settings import *


class ImageGenerationApp:
    def __init__(self, root, model_weights_path, model_gan, args):
        self.root = root
        self.root.title("GAN Image Generator")
        self.weights_loaded = model_weights_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model_gan
        self.current_mask = None
        self.args = args
        self.current_mask_path = None
        
        self.setup_ui()

    def load_custom_weights(self):
        """Ручная загрузка пользовательских весов"""
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")]
        )
        if not path:
            return
            
        try:
            self.model.load_checkpoint(os.path.dirname(path))
            self.weights_loaded = True
            self.log(f"✅ Custom weights loaded from: {path}")
        except Exception as e:
            self.log(f"❌ Error loading weights: {str(e)}")
            messagebox.showerror("Error", f"Failed to load weights: {str(e)}")

    def setup_ui(self):
        # Загрузка
        load_frame = tk.Frame(self.root)
        load_frame.pack(pady=10)

        tk.Button(load_frame, text="Load Custom Weights", command=self.load_custom_weights).grid(row=0, column=0, padx=5)
        tk.Button(load_frame, text="Load Binary Mask", command=self.load_mask).grid(row=0, column=1, padx=5)

        # Отображение
        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=10)

        tk.Label(display_frame, text="Input Image").grid(row=0, column=0)
        tk.Label(display_frame, text="Generated Image").grid(row=0, column=1)

        self.mask_label = tk.Label(display_frame)
        self.mask_label.grid(row=1, column=0, padx=10)

        self.result_label = tk.Label(display_frame)
        self.result_label.grid(row=1, column=1, padx=10)

        # Кнопка генерации
        tk.Button(self.root, text="Generate Image", command=self.generate_image, 
                height=2, width=20).pack(pady=20)

        # Логи
        self.log_box = tk.Text(self.root, height=8, width=80)
        self.log_box.pack(padx=10, pady=10)

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_box.insert(tk.END, f"{timestamp} {message}\n")
        self.log_box.see(tk.END)

    def load_mask(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.tif"), ("All Files", "*.*")])
        if path:
            try:
                self.current_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                self.current_mask_path = path
                mask = Image.fromarray(self.current_mask)
                mask = mask.resize((256, 256))
                photo = ImageTk.PhotoImage(mask)
                self.mask_label.config(image=photo)
                self.mask_label.image = photo
                self.log("📥 Mask loaded successfully")
            except Exception as e:
                self.log(f"❌ Error loading mask: {str(e)}")
                messagebox.showerror("Error", f"Failed to load mask: {str(e)}")

    def generate_image(self):
        if not self.weights_loaded:
            messagebox.showerror("Ошибка", "Веса модели не загружены!")
            return

            
        
        if self.current_mask is None:
            messagebox.showerror("Ошибка", "Загрузите маску!")
            return

        try:
            
            preprocessor = IceRidgeDatasetPreprocessor(PREPROCESSORS)
            preprocessed_img = preprocessor.process_image(self.current_mask)
            
            processor = InferenceMaskingProcessor(outpaint_ratio=0.2)
            generated, _ = self.model.infer_generate(preprocessed_img=preprocessed_img, 
                                                        checkpoint_path=WEIGHTS_PATH, 
                                                        processor=processor)

            mask_dir = os.path.dirname(self.current_mask_path)
            mask_name = os.path.splitext(os.path.basename(self.current_mask_path))[0]
            output_path = os.path.join(mask_dir, f"{mask_name}_generated.png")
            

            original_shape = self.current_mask.shape
            resized_generated = cv2.resize(generated, (int(original_shape[1] * 1.2), int(original_shape[0] * 1.2)), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(output_path, resized_generated)

            self.log(f"✅ Success! Result saved to: {output_path}")
            print(f"Генерация завершена. Результат сохранён в {output_path}")

            result_img = Image.open(output_path).resize((256, 256))
            result_photo = ImageTk.PhotoImage(result_img)
            self.result_label.config(image=result_photo)
            self.result_label.image = result_photo 


        except Exception as e:
            self.log(f"❌ Generation error: {str(e)}")
            messagebox.showerror("Error", f"Generation failed: {str(e)}")