import sys
import traceback

from PyQt5.QtWidgets import QApplication, QMessageBox

from modules.chat_window import ChatWindow


def exception_hook(exctype, value, traceback):
    """全局异常处理函数"""
    error_msg = f"未捕获的异常: {exctype.__name__}: {value}"
    print(error_msg)

    # 尝试将错误写入日志文件
    try:
        with open("error_log.txt", "a") as f:
            import datetime
            f.write(f"[{datetime.datetime.now()}] {error_msg}\n")
            import traceback as tb
            tb.print_exception(exctype, value, traceback, file=f)
    except:
        pass

    # 调用原始的异常处理器
    sys.__excepthook__(exctype, value, traceback)


def main():
    app = QApplication(sys.argv)
    # 设置全局异常处理器
    sys.excepthook = exception_hook
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
