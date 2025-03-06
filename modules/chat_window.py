import sys
import json
import time
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QComboBox, QSlider, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QIcon

from modules.ollama_client import OllamaClient


class ChatThread(QThread):
    response_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, client, model, messages, temperature, max_tokens):
        super().__init__()
        self.client = client
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            stream_response = self.client.chat_completion(
                model=self.model,
                messages=self.messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            for chunk in stream_response:
                if self._stop:
                    break
                if chunk:
                    content_delta = chunk["choices"][0]["delta"].get("content", "")
                    if content_delta:
                        self.response_signal.emit(content_delta)
        except Exception as e:
            self.response_signal.emit(f"\n\n发生错误: {str(e)}")
        finally:
            self.finished_signal.emit()


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = OllamaClient()
        self.messages = [{"role": "system", "content": "你是一个有用的AI助手。"}]
        self.current_response = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ollama 本地聊天')
        self.setWindowIcon(QIcon("./imgs/crow_TqT_icon.ico"))
        self.setGeometry(100, 100, 1366, 768)  # 修改为1024*768

        # 创建主布局
        main_layout = QVBoxLayout()

        # 创建设置区域
        settings_layout = QHBoxLayout()

        # 服务器地址
        settings_layout.addWidget(QLabel("服务器地址:"))
        self.server_input = QLineEdit("http://192.168.1.114:11434")
        settings_layout.addWidget(self.server_input)

        # 模型选择
        settings_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["qwq:latest", "llama3", "mixtral", "phi3"])
        settings_layout.addWidget(self.model_combo)

        # 温度
        settings_layout.addWidget(QLabel("温度:"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)  # 默认0.7
        self.temp_slider.setFixedWidth(100)
        settings_layout.addWidget(self.temp_slider)
        self.temp_label = QLabel("0.7")
        settings_layout.addWidget(self.temp_label)
        self.temp_slider.valueChanged.connect(self.update_temp_label)

        # 最大token
        settings_layout.addWidget(QLabel("最大Token:"))
        self.token_combo = QComboBox()
        self.token_combo.addItems(["1024", "2048", "4096", "8192"])
        self.token_combo.setCurrentIndex(2)  # 默认2048
        settings_layout.addWidget(self.token_combo)

        main_layout.addLayout(settings_layout)

        # 创建聊天窗口和输入区域的垂直布局
        chat_input_layout = QVBoxLayout()

        # 聊天历史
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 10))
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
            }
        """)
        chat_input_layout.addWidget(self.chat_history, 1)  # 分配更多空间给聊天历史

        # 用户输入区域
        input_layout = QHBoxLayout()
        self.user_input = QTextEdit()
        self.user_input.setFixedHeight(100)
        self.user_input.setFont(QFont("Arial", 10))
        self.user_input.setPlaceholderText("在此输入您的问题...")
        self.user_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        """)
        self.user_input.installEventFilter(self)  # 安装事件过滤器

        button_layout = QVBoxLayout()
        send_button = QPushButton("发送")
        send_button.clicked.connect(self.send_message)
        clear_button = QPushButton("清空对话")
        clear_button.clicked.connect(self.clear_chat)
        clear_context_button = QPushButton("清除上下文")
        clear_context_button.clicked.connect(self.clear_context)

        # 添加一个暂停/继续按钮
        self.stop_button = QPushButton("暂停")
        self.stop_button.setEnabled(False)  # 初始禁用
        self.stop_button.clicked.connect(self.toggle_generation)

        button_layout.addWidget(send_button)
        button_layout.addWidget(clear_button)
        button_layout.addWidget(clear_context_button)
        button_layout.addWidget(self.stop_button)

        input_layout.addWidget(self.user_input)
        input_layout.addLayout(button_layout)

        # 将输入区域添加到聊天输入布局的底部
        chat_input_layout.addLayout(input_layout)

        main_layout.addLayout(chat_input_layout)

        # 设置主窗口的中央组件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 设置样式表
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
            }
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                min-width: 80px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a80d2;
            }
            QPushButton:disabled {
                background-color: #9eb8d6;
            }
            QComboBox, QLineEdit {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 8px;
                background-color: white;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 8px;
                background: #e0e0e0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #505050;
            }
        """)

    def clear_context(self):
        # 只保留系统提示，不清空聊天界面
        self.messages = [{"role": "system", "content": "你是一个有用的AI助手。"}]
        self.chat_history.append('<div style="color:#888;"><i>--- 上下文已清除，但对话历史保留 ---</i></div>')

    # 添加控制方法
    def toggle_generation(self):
        if hasattr(self, 'chat_thread') and self.chat_thread.isRunning():
            self.chat_thread.stop()
            self.stop_button.setText("已暂停")
            self.stop_button.setEnabled(False)

    def update_temp_label(self):
        value = self.temp_slider.value() / 100
        self.temp_label.setText(f"{value:.1f}")

    def eventFilter(self, obj, event):
        if obj == self.user_input and event.type() == event.KeyPress:
            # 检查是否按下回车键
            if event.key() == Qt.Key_Return:
                # 如果同时按下Shift键，则插入换行
                if event.modifiers() & Qt.ShiftModifier:
                    return False  # 让系统处理这个事件（添加换行）
                else:
                    # 没有按Shift，直接发送消息
                    self.send_message()
                    return True  # 我们已经处理了这个事件
        return super().eventFilter(obj, event)

    def send_message(self):
        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            return

        # 更新服务器地址
        self.client.base_url = self.server_input.text().strip()

        # 添加用户消息到历史
        self.chat_history.append(f'<div style="color:blue;"><b>你:</b> {user_text}</div>')
        self.messages.append({"role": "user", "content": user_text})

        # 清空输入框
        self.user_input.clear()

        # 显示思考中
        self.chat_history.append('<div style="color:green;"><b>AI助手:</b> </div>')
        self.current_response = ""

        # 准备参数
        model = self.model_combo.currentText()
        temperature = self.temp_slider.value() / 100
        max_tokens = int(self.token_combo.currentText())

        # 启动线程处理请求
        self.chat_thread = ChatThread(
            self.client,
            model,
            self.messages.copy(),
            temperature,
            max_tokens
        )
        self.chat_thread.response_signal.connect(self.update_response)
        self.chat_thread.finished_signal.connect(self.complete_response)
        self.chat_thread.start()

        # 启用暂停按钮
        self.stop_button.setText("暂停")
        self.stop_button.setEnabled(True)

    @pyqtSlot(str)
    def update_response(self, text):
        # 更新当前响应（仅累积原始文本）
        self.current_response += text

        # 找到最后一个AI回复并更新为当前累积的原始内容
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)

        # 替换为更新的内容（不做任何格式化处理）
        cursor.removeSelectedText()
        cursor.insertHtml(f'<div style="color:green;"><b>AI助手:</b> {self.current_response}</div>')

        # 滚动到底部
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)

    @pyqtSlot()
    def complete_response(self):
        # 响应完成，处理最终格式并更新消息历史
        # print(self.current_response)
        # 处理思考过程与最终回答的区分
        import re
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        think_match = think_pattern.search(self.current_response)

        # 找到最后一个AI回复并准备更新
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

        # 创建最终要展示的内容
        if think_match:
            # 提取思考过程
            think_content = think_match.group(1).strip()
            # 获取思考后的实际回答（移除think标签及其内容）
            actual_response = re.sub(r'<think>.*?</think>', '', self.current_response, flags=re.DOTALL).strip()

            # 构建最终显示内容 - 思考过程小字体灰色，实际回答正常大小绿色
            final_display = f'''
            <div style="color:#888; font-size:12px; background-color:#f0f0f0; padding:5px; border-radius:5px; margin-bottom:10px;">
                <i>思考过程:</i><br>{think_content}
            </div>
            <div style="color:green; margin-top:14px;">
                {actual_response}
            </div>
            '''
        else:
            # 没有思考过程，直接显示回答
            final_display = self.current_response

        # 更新聊天窗口显示
        cursor.insertHtml(f'<div style="color:green;"><b>AI助手:</b> {final_display}</div>')

        # 添加AI回复到消息历史（存储原始内容）
        self.messages.append({"role": "assistant", "content": self.current_response})

        # 禁用暂停按钮
        self.stop_button.setEnabled(False)
        self.stop_button.setText("暂停")

    def clear_chat(self):
        # 清空聊天历史
        self.chat_history.clear()
        self.messages = [{"role": "system", "content": "你是一个有用的AI助手。"}]
        self.current_response = ""


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())