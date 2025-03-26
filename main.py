import window

import sys
import PySide6.QtWidgets as QtW

if __name__ == "__main__":
    app = QtW.QApplication(sys.argv)
    window = window.Alprs()
    window.show()
    sys.exit(app.exec())
