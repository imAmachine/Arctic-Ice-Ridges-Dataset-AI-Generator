import tkinter as tk
import os

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime

from src.models.inference import InferenceManager


class ImageGenerationApp:
    def __init__(self, root, config):
        self.root = root
        self.root.title("Генератор изображений")
        self.config = config
        self.generator = InferenceManager(config)
        
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

    def _create_log_box(self):
        self.log_box = tk.Text(self.root, height=8, width=80)
        self.log_box.pack(padx=10, pady=10)  

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_box.insert(tk.END, f"{timestamp} {message}\n")
        self.log_box.see(tk.END)

    def select_model(self, choice):
        try:
            self.generator.load_model(choice)
            self.log(f"🧠 Модель {choice} инициализирована.")
        except Exception as e:
            self.log(f"❌ Ошибка инициализации модели: {e}")
            messagebox.showerror("Ошибка", str(e))

    def load_custom_weights(self):
        """Ручная загрузка пользовательских весов"""
        path = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.generator.load_weights(path)
            self.log(f"✅ Пользовательские веса, загруженные из: {path}")
        except Exception as e:
            self.log(f"❌ Ошибка загрузки весов: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить веса: {str(e)}")

    def load_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.tif"), ("All Files", "*.*")])
        if path:
            try:
                mask = self.generator.load_mask(path)
                photo = ImageTk.PhotoImage(mask)
                self._update_image_label(self.mask_label, mask)
                self.mask_label.config(image=photo)
                self.mask_label.image = photo
                self.log("📥 Маска успешно загружена")
            except Exception as e:
                self.log(f"❌ Ошибка загрузки маски.: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить маску: {str(e)}")

    def generate_image(self):
        if not self._check_generation_conditions():
            return
        try:
            result = self.generator.generate()
            self._update_image_label(self.result_label, result)
            self.log("✅ Генерация завершена")
        except Exception as e:
            self.log(f"❌ Ошибка генерации: {e}")
            messagebox.showerror("Ошибка", str(e))

    def save_image(self):
        if self.generator.last_generated_image is None:
            messagebox.showwarning("Внимание", "Сначала сгенерируйте изображение.")
            return
        
        if self.generator.current_mask_path:
            base_name = os.path.splitext(os.path.basename(self.generator.current_mask_path))[0] + "_generated"
        else:
            base_name = "generated"

        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{base_name}.png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if not path:
            return

        try:
            self.generator.save_last_image(path)
            self.log(f"💾 Изображение сохранено в: {path}")
        except Exception as e:
            self.log(f"❌ Ошибка при сохранении: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {str(e)}")

    def _update_image_label(self, label, image: Image.Image):
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def _check_generation_conditions(self) -> bool:
        if not self.generator.weights_loaded:
            messagebox.showerror("Ошибка", "Веса модели не загружены!")
            return False
        if self.generator.current_mask is None:
            messagebox.showerror("Ошибка", "Загрузите маску!")
            return False
        if self.generator.model is None:
            messagebox.showerror("Ошибка", "Выберете модель!")
            return False
        return True