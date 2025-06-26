from PyQt5.QtWidgets import QApplication, QWidget

import sys

from Z_UI_try_ui import Ui_Form

class mainUI(Ui_Form, QWidget):
    def __init__(self, parent=None):
        super(mainUI, self).__init__(parent)
        self.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainUI()
    window.show()
    sys.exit(app.exec())