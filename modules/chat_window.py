import os
import re
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


def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和PyInstaller打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 创建了临时文件夹，将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


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
        self.setWindowTitle('JR-Ollama Chat')
        self.setWindowIcon(QIcon(resource_path("imgs/crow_TqT_icon.ico")))
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
                background-color: #1e1e1e;  /* 更深的灰黑色 */
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
            }
        """)
        # 添加这一行以确保文本自动换行
        self.chat_history.setLineWrapMode(QTextEdit.WidgetWidth)
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

        # 设置样式表 - 修改为更深的灰黑色主题
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTextEdit {
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                background-color: #2d2d2d;
                color: #e0e0e0;
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
                background-color: #555555;
            }
            QComboBox, QLineEdit {
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px;
                background-color: #333333;
                color: #e0e0e0;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #e0e0e0;
                selection-background-color: #4a90e2;
            }
            QSlider::groove:horizontal {
                border: none;
                height: 8px;
                background: #444444;
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
                color: #c0c0c0;
            }
        """)

    def clear_context(self):
        # 只保留系统提示，不清空聊天界面
        self.messages = [{"role": "system", "content": "你是一个有用的AI助手。"}]
        self.chat_history.append('<div style="color:#808080;"><i>--- 上下文已清除，但对话历史保留 ---</i></div>')

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

        # 添加用户消息到历史 - 修改为浅蓝色
        self.chat_history.append(f'<div style="color:#7ebeff;"><b>你:</b> {user_text}</div>')
        self.messages.append({"role": "user", "content": user_text})

        # 清空输入框
        self.user_input.clear()

        # 显示思考中
        self.chat_history.append('<div style="color:#e0e0e0;"><b>AI助手:</b> </div>')
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
        cursor.insertHtml(
            f'<div style="color:#e0e0e0; word-wrap: break-word; max-width: 100%;"><b>AI助手:</b> {self.current_response}</div>')

        # 滚动到底部
        cursor.movePosition(QTextCursor.End)
        self.chat_history.setTextCursor(cursor)

    @pyqtSlot()
    def complete_response(self):
        import re
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        think_match = think_pattern.search(self.current_response)

        # 找到最后一个AI回复并准备更新
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()

        # 获取格式化后的响应文本
        if think_match:
            # 提取思考过程
            think_content = think_match.group(1).strip()
            # 获取思考后的实际回答（移除think标签及其内容）
            actual_response = re.sub(r'<think>.*?</think>', '', self.current_response, flags=re.DOTALL).strip()

            # 处理思考过程中的格式
            think_content = self.format_markdown_and_code(think_content)

            # 处理实际回答中的格式
            formatted_response = self.format_markdown_and_code(actual_response)

            # 构建最终显示内容，添加自动换行
            final_display = f'''
            <div style="color:#a0a0a0; font-size:15px; background-color:#333333; padding:5px; border-radius:5px; margin-bottom:10px; word-wrap: break-word; max-width: 100%;">
                <i>思考过程:</i><br>{think_content}
            </div>
            <div style="color:#e0e0e0; margin-top:14px; word-wrap: break-word; max-width: 100%;">
                {formatted_response}
            </div>
            '''

            # 保存应该是实际回答，而不是带思考过程的完整内容
            response_for_history = actual_response
        else:
            # 没有思考过程，直接显示格式化后的回答
            formatted_response = self.format_markdown_and_code(self.current_response)
            final_display = formatted_response
            response_for_history = self.current_response

        # 更新聊天窗口显示，确保正确换行
        cursor.insertHtml(
            f'<div style="color:#e0e0e0; word-wrap: break-word; max-width: 100%;"><b>AI助手:</b> {final_display}</div>')

        # 添加模型的回答到历史消息列表中 - 这是解决问题的关键
        self.messages.append({"role": "assistant", "content": response_for_history})

    def format_markdown_and_code(self, text):
        """格式化文本中的Markdown和代码片段"""
        import re

        # 先保存代码块，防止它们被其他规则干扰
        code_blocks = []

        def save_code_block(match):
            # 获取整个匹配内容以保留原始格式
            full_match = match.group(0)

            # 尝试提取语言标识符和代码内容
            lines = full_match.strip().split('\n')

            if len(lines) >= 2 and lines[0].startswith('```'):
                # 如果是标准格式的代码块
                language = lines[0][3:].strip()
                code_content = '\n'.join(lines[1:-1])  # 除去首尾行
            else:
                # 如果是不标准的格式（如只有语言标识符没有```）
                language = lines[0].strip()
                code_content = '\n'.join(lines[1:])

            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"

            # 处理代码高亮
            highlighted_code = self.syntax_highlight(code_content, language) if language else code_content.replace("<",
                                                                                                                   "&lt;").replace(
                ">", "&gt;")

            # 创建代码块HTML，使用统一黑色背景
            html = f'''<div style="background-color: #282c34; border-radius: 4px; margin: 8px 0; overflow: auto;">
    <pre style="margin: 0; padding: 10px; font-family: 'Source Code Pro', Consolas, 'Courier New', monospace; line-height: 1.5; overflow-x: auto; white-space: pre; color: #abb2bf;">{highlighted_code}</pre>
    </div>'''

            code_blocks.append(html)
            return placeholder

        # 使用更广泛的模式匹配代码块，包括非标准格式
        # 这个模式会匹配：1) ```语言 代码 ``` 2) 语言名 + 缩进代码
        code_block_pattern = r'(```\w*[\s\S]*?```|(?:^|\n)(?:python|java|javascript|bash|shell|cmd|cpp|csharp|c\+\+|c#|html|css|php|go|rust|swift|ruby|typescript|ts|js)\s*\n\s{4}.*?(?=\n\S|$))'
        text = re.sub(code_block_pattern, save_code_block, text, flags=re.MULTILINE | re.DOTALL)

        # 行内代码处理
        inline_codes = []

        def save_inline_code(match):
            code = match.group(1)
            placeholder = f"__INLINE_CODE_{len(inline_codes)}__"
            html = f'<code style="background-color: #2c323c; padding: 2px 4px; border-radius: 3px; font-family: monospace; color: #abb2bf;">{code.replace("<", "&lt;").replace(">", "&gt;")}</code>'
            inline_codes.append(html)
            return placeholder

        # 处理行内代码
        text = re.sub(r'`([^`]+)`', save_inline_code, text)

        # 处理连续换行问题
        text = re.sub(r'\n{2,}', '\n', text)

        # 删除文档末尾的破折号分隔符
        text = re.sub(r'\n---\s*$', '', text)

        # 将换行符转换为HTML换行
        text = text.replace('\n', '<br>')

        # 处理其他Markdown格式
        # 处理标题 - 修改为白色
        text = re.sub(r'(#{1,6})\s+(.*?)(?:<br>|$)',
                      lambda
                          m: f'<h{len(m.group(1))} style="font-size: {max(18 - len(m.group(1)) * 2, 12)}px; font-weight: bold; margin: 10px 0; color: #ffffff;">{m.group(2)}</h{len(m.group(1))}>',
                      text)

        # 处理粗体和斜体 - 确保显示为白色
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong style="color: #ffffff;">\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em style="color: #ffffff;">\1</em>', text)

        # 确保普通文本也是白色
        text = f'<span style="color: #e0e0e0;">{text}</span>'

        # 最后，恢复代码块和行内代码
        for i, html in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", html)

        for i, html in enumerate(inline_codes):
            text = text.replace(f"__INLINE_CODE_{i}__", html)

        return text

    def syntax_highlight(self, code, language):
        """为代码添加语法高亮"""
        code = code.replace("<", "&lt;").replace(">", "&gt;")

        # 通用关键字和颜色定义
        colors = {
            'keyword': '#c678dd',  # 关键字为紫色
            'string': '#98c379',  # 字符串为绿色
            'number': '#d19a66',  # 数字为橙色
            'comment': '#5c6370',  # 注释为灰色
            'function': '#61afef',  # 函数为蓝色
            'constant': '#e06c75',  # 常量为红色
            'operator': '#56b6c2',  # 运算符为青色
            'property': '#e6c07b',  # 属性为黄色
        }

        if language.lower() in ['python', 'py']:
            # Python高亮规则
            keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
                        'import', 'from', 'as', 'return', 'yield', 'with', 'lambda', 'not', 'in',
                        'and', 'or', 'True', 'False', 'None', 'self', 'pass', 'break', 'continue']

            for keyword in keywords:
                code = re.sub(r'\b(' + keyword + r')\b',
                              f'<span style="color: {colors["keyword"]}">\\1</span>', code)

            # 字符串高亮 (单引号和双引号)
            code = re.sub(r'(\'[^\']*\'|\"[^\"]*\")',
                          f'<span style="color: {colors["string"]}">\\1</span>', code)

            # 数字高亮
            code = re.sub(r'\b(\d+\.?\d*)\b',
                          f'<span style="color: {colors["number"]}">\\1</span>', code)

            # 注释高亮
            code = re.sub(r'(#.*?)(?=<br>|$)',
                          f'<span style="color: {colors["comment"]}">\\1</span>', code)

            # 函数调用
            code = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\(',
                          f'<span style="color: {colors["function"]}">\\1</span>(', code)

            # 装饰器
            code = re.sub(r'(@[a-zA-Z_][a-zA-Z0-9_]*)',
                          f'<span style="color: {colors["property"]}">\\1</span>', code)

        elif language.lower() in ['javascript', 'js']:
            # JavaScript高亮规则
            keywords = ['var', 'let', 'const', 'function', 'return', 'if', 'else', 'for', 'while',
                        'class', 'new', 'this', 'import', 'export', 'from', 'try', 'catch', 'finally',
                        'switch', 'case', 'break', 'continue', 'default', 'null', 'undefined', 'true', 'false']

            for keyword in keywords:
                code = re.sub(r'\b(' + keyword + r')\b',
                              f'<span style="color: {colors["keyword"]}">\\1</span>', code)

            # 字符串高亮
            code = re.sub(r'(\'[^\']*\'|\"[^\"]*\"|`[^`]*`)',
                          f'<span style="color: {colors["string"]}">\\1</span>', code)

            # 数字高亮
            code = re.sub(r'\b(\d+\.?\d*)\b',
                          f'<span style="color: {colors["number"]}">\\1</span>', code)

            # 注释高亮
            code = re.sub(r'(//.*?)(?=<br>|$)|(/\*.*?\*/)',
                          f'<span style="color: {colors["comment"]}">\\1</span>', code)

            # 函数调用
            code = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\(',
                          f'<span style="color: {colors["function"]}">\\1</span>(', code)

        elif language.lower() in ['html']:
            # HTML高亮规则
            # 标签
            code = re.sub(r'(&lt;/?[a-z][a-z0-9]*(?:\s+[a-z0-9\-]+(?:=(?:\".*?\"|\'.*?\'|[^\s&gt;]*))?)*\s*/?\s*&gt;)',
                          f'<span style="color: {colors["keyword"]}">\\1</span>', code)

            # 属性
            code = re.sub(r'(\s+)([a-z0-9\-]+)(=)',
                          f'\\1<span style="color: {colors["property"]}">\\2</span>\\3', code)

            # 属性值
            code = re.sub(r'(=)(\".*?\"|\'.*?\')',
                          f'\\1<span style="color: {colors["string"]}">\\2</span>', code)

        return code

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
