import sys
from PyQt5.QtWidgets import QApplication

from modules.chat_window import ChatWindow


def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
