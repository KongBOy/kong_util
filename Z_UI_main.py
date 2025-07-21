from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QLineEdit

import sys

from Z_UI_try_ui import Ui_Form

from pdf2jpg import Convert_dir_pdf_to_jpg

class mainUI(Ui_Form, QWidget):
    def __init__(self, parent=None):
        super(mainUI, self).__init__(parent)
        self.setupUi(self)


        self.src_dir_get_file_fun = lambda: self.get_files(self.lineEdit)
        self.dst_dir_get_file_fun = lambda: self.get_files(self.lineEdit_2)

        self.pushButton_2.clicked.connect(self.src_dir_get_file_fun)
        self.pushButton_3.clicked.connect(self.dst_dir_get_file_fun)

        self.pushButton.clicked.connect(self.run)

    def get_files(self, line_edit:QLineEdit):
        file_dialog = QFileDialog()
        dir_name = file_dialog.getExistingDirectory()
        line_edit.setText(dir_name)

    def run(self):
        if(self.lineEdit.text() == "")  : self.src_dir_get_file_fun()
        if(self.lineEdit_2.text() == "")  : self.lineEdit_2.setText(  self.lineEdit.text() + "/pdf_to_jpg_result" )


        Convert_dir_pdf_to_jpg(self.lineEdit.text(), self.lineEdit_2.text(), print_msg=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = mainUI()
    window.show()
    sys.exit(app.exec())