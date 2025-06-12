import sys
from typing import Dict

from PyQt5 import QtWidgets

from src.gui.main_window import MainWindow


class AppStart:
    def __init__(self, config_serializer: Dict):
        self.config_serializer = config_serializer

    def start(self):
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(self.config_serializer)
        window.show()
        sys.exit(app.exec())