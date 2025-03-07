import sys
import traceback

from PyQt5.QtWidgets import QApplication, QMessageBox

from modules.chat_window import ChatWindow


def main():
    app = QApplication(sys.argv)

    try:
        window = ChatWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        error_msg = f"程序发生错误:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)  # 输出到控制台

        # 显示错误对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("程序错误")
        msg_box.setText(error_msg)
        msg_box.exec_()

        sys.exit(1)


if __name__ == "__main__":
    main()
