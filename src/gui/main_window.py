import os

from PyQt5 import QtWidgets, uic

from src.gui.inference_window import InferenceWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.interfaces = './src/gui/interfaces'
        path = os.path.join(self.interfaces, 'main_window.ui')
        uic.loadUi(path, self)
        
        self.config = config
        
        self.inference_window = None
        self._setup_ui()
    
    def _setup_ui(self):
        self.inference_button.clicked.connect(self.open_inference)
        self.train_button.clicked.connect(self.open_train)
    
    def open_inference(self):
        """Открывает окно инференса"""
        if self.inference_window is None:
            self.inference_window = InferenceWindow(self.config, self.interfaces, parent=self)
        self.inference_window.show()
        self.hide()

    def open_train(self):
        pass
    
    def show_main(self):
        """Возвращает в главное окно"""
        self.show()
        if self.inference_window:
            self.inference_window.hide()