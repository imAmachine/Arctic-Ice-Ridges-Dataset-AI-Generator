import os

from datetime import datetime
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from io import BytesIO

from src.models.inference import InferenceManager

class InferenceWindow(QtWidgets.QMainWindow):
    def __init__(self, config, interfaces):
        super().__init__()
        path = os.path.join(interfaces, 'inference.ui')
        uic.loadUi(path, self)

        self.generator = InferenceManager(config)

        self._setup_ui()

    def _setup_ui(self):
        self.load_image.setScaledContents(True)
        self.generated_image.setScaledContents(True)

        self.load_weight_button.clicked.connect(self.load_custom_weights)
        self.load_mask_button.clicked.connect(self.load_mask)
        self.Generation_image.clicked.connect(self.generate_image)
        self.save_button.clicked.connect(self.save_image)

        self.comboBox.setCurrentIndex(-1)
        self.comboBox.currentTextChanged.connect(self.select_model)

    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log.append(f"{timestamp} {message}")

    def select_model(self, choice):
        try:
            self.generator.load_model(choice)
            self.log.append(f"🧠 Модель {choice} инициализирована.")
        except Exception as e:
            self.log.append(f"❌ Ошибка инициализации модели: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def load_custom_weights(self):
        """Ручная загрузка пользовательских весов"""
        path, _ = QFileDialog.getOpenFileName(self, "Выберите веса модели", "", "(*.pt);;All Files (*)")
        if not path:
            return
        try:
            self.generator.load_weights(path)
            self.log.append(f"✅ Пользовательские веса загружены из: {path}")
        except Exception as e:
            self.log.append(f"❌ Ошибка загрузки весов: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить веса:\n{str(e)}")

    def load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение маски", "", "(*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if not path:
            return
        try:
            pil_image = self.generator.load_mask(path)

            buf = BytesIO()
            pil_image.save(buf, format='PNG')
            buf.seek(0)

            qimage = QImage()
            qimage.loadFromData(buf.read(), 'PNG')
            pixmap = QPixmap.fromImage(qimage)

            self.load_image.setPixmap(pixmap)
            self.log.append(f"📥 Маска успешно загружена")

        except Exception as e:
            QMessageBox.critical(self, f"❌ Ошибка загрузки маски.: {str(e)}")
            self.log.append("Ошибка", f"Не удалось загрузить маску: {str(e)}")

    def generate_image(self):
        """Генерация изображения"""
        if not self._check_generation_conditions():
            return
            
        try:
            result = self.generator.generate()
            buf = BytesIO()
            result.save(buf, format='PNG')
            buf.seek(0)

            qimage = QImage()
            qimage.loadFromData(buf.read(), 'PNG')
            self.generated_image.setPixmap(QPixmap.fromImage(qimage))
            self.log.append("✅ Генерация завершена")
        except Exception as e:
            self.log.append(f"❌ Ошибка генерации: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def save_image(self):
        """Сохранение сгенерированного изображения"""
        if self.generator.last_generated_image is None:
            QMessageBox.warning(self, "Внимание", "Сначала сгенерируйте изображение.")
            return
        
        if self.generator.current_mask_path:
            base_name = os.path.splitext(os.path.basename(self.generator.current_mask_path))[0] + "_generated"
        else:
            base_name = "generated"

        path, _ = QFileDialog.getSaveFileName(self, "Сохранение изображения", base_name, "PNG Image (*.png);;All Files (*)")
        if not path:
            return
        
        try:
            self.generator.save_last_image(path)
            self.log.append(f"💾 Изображение сохранено в: {path}")
        except Exception as e:
            self.log.append(f"❌ Ошибка при сохранении: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение:\n{str(e)}")

    def _check_generation_conditions(self):
        """Проверка условий перед генерацией"""
        if not self.generator.weights_loaded:
            QMessageBox.critical(self, "Ошибка", "Веса модели не загружены!")
            return False
        if self.generator.current_mask is None:
            QMessageBox.critical(self, "Ошибка", "Загрузите маску!")
            return False
        if self.generator.model is None:
            QMessageBox.critical(self, "Ошибка", "Выберите модель!")
            return False
        return True