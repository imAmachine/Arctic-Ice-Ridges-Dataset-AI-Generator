import sys
import torch

from PyQt5 import QtWidgets

from generativelib.config_tools.base import ConfigReader
from src.gui.main_window import MainWindow


class AppStart:
    def __init__(self, config_deserializer: ConfigReader):
        self.config_serializer = config_deserializer

    def start(self, device: torch.device):
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(self.config_serializer, device)
        window.show()
        sys.exit(app.exec())