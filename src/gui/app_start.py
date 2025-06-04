import sys

from PyQt5 import QtWidgets
from src.config_wrappers import TrainConfigSerializer

from gui.main_window import MainWindow


class AppStart:
    def __init__(self, config_serializer: TrainConfigSerializer):
        self.config_serializer = config_serializer

    def start(self):
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow(self.config_serializer, section='path', keys='interfaces')
        window.show()
        sys.exit(app.exec())