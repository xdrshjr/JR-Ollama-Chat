import os
import re
import sys
import json
import time
import requests
from langchain.vectorstores import faiss

from modules.chat_thread import ChatThread
from modules.ollama_memory_manager import OllamaMemoryManager
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QComboBox, QSlider, QSplitter, QCheckBox,
                             QGroupBox, QFrame, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor, QIcon

from modules.ollama_client import OllamaClient
from modules.thought_generator import ThoughtGenerator


def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和PyInstaller打包后的环境 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 创建了临时文件夹，将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = OllamaClient()
        self.messages = [{"role": "system", "content": "你是一个有用的AI助手。"}]
        self.current_response = ""
        self.is_thinking = False
        self.thought_generator = None

        # 检查记忆目录
        memory_dir = "memory"
        import os
        if os.path.exists(memory_dir) and os.path.isdir(memory_dir):
            memory_file = os.path.join(memory_dir, "memories.json")
            if os.path.exists(memory_file):
                import json
                try:
                    with open(memory_file, "r", encoding="utf-8") as f:
                        memories = json.load(f)
                    self.has_existing_memories = len(memories) > 0
                    self.memory_count_startup = len(memories)
                except:
                    self.has_existing_memories = False
                    self.memory_count_startup = 0
            else:
                self.has_existing_memories = False
                self.memory_count_startup = 0
        else:
            self.has_existing_memories = False
            self.memory_count_startup = 0

        self.initUI()

        # 记录启动信息
        self.log_to_console("JR-Ollama Chat 已启动")
        self.log_to_console(f"服务器地址: {self.server_input.text()}")
        self.log_to_console(f"默认模型: {self.model_combo.currentText()}")

        # 如果有现有记忆，显示通知
        if self.has_existing_memories:
            self.chat_history.append(
                f'<div style="color:#808080;"><i>--- 检测到 {self.memory_count_startup} 条已存储的记忆，可勾选"使用记忆"启用 ---</i></div>')

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

        # 嵌入模型选择
        settings_layout.addWidget(QLabel("嵌入模型:"))
        self.embed_model_combo = QComboBox()
        self.embed_model_combo.addItems(["bge-m3"])
        self.embed_model_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.embed_model_combo)
        self.embed_model_combo.currentTextChanged.connect(self.update_embedding_model)

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

        # 添加记忆功能控制部分
        memory_layout = QHBoxLayout()

        self.use_memory_cb = QCheckBox("使用记忆")
        self.use_memory_cb.setChecked(False)
        self.use_memory_cb.setStyleSheet("color: #e0e0e0;")
        self.use_memory_cb.stateChanged.connect(self.toggle_memory_usage)  # 添加事件处理
        memory_layout.addWidget(self.use_memory_cb)

        self.memory_status = QLabel("记忆状态: 空闲")
        self.memory_status.setStyleSheet("color: #e0e0e0;")
        memory_layout.addWidget(self.memory_status)

        # 修改为开关按钮
        self.thinking_toggle = QCheckBox("自我思考")
        self.thinking_toggle.setChecked(False)
        self.thinking_toggle.clicked.connect(self.toggle_thinking)
        memory_layout.addWidget(self.thinking_toggle)

        self.memory_count = QLabel("记忆数量: 0")
        self.memory_count.setStyleSheet("color: #e0e0e0;")
        memory_layout.addWidget(self.memory_count)

        # 记忆相关度阈值
        memory_layout.addWidget(QLabel("记忆相关度阈值:"))
        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setRange(0, 100)
        self.similarity_threshold_slider.setValue(65)  # 默认0.65
        self.similarity_threshold_slider.setFixedWidth(100)
        memory_layout.addWidget(self.similarity_threshold_slider)
        self.similarity_threshold_label = QLabel("0.65")
        memory_layout.addWidget(self.similarity_threshold_label)
        self.similarity_threshold_slider.valueChanged.connect(self.update_similarity_threshold_label)

        # 在initUI的记忆布局中添加
        self.memory_stats_btn = QPushButton("记忆统计")
        self.memory_stats_btn.clicked.connect(self.show_memory_statistics)
        memory_layout.addWidget(self.memory_stats_btn)

        # 删除所有记忆按钮
        self.clear_memory_btn = QPushButton("清空所有记忆")
        self.clear_memory_btn.clicked.connect(self.clear_all_memories)
        memory_layout.addWidget(self.clear_memory_btn)

        main_layout.addLayout(memory_layout)

        # 创建分割显示区域
        self.splitter = QSplitter(Qt.Horizontal)

        # 左侧聊天区域 (3/4)
        self.chat_widget = QWidget()
        chat_layout = QVBoxLayout(self.chat_widget)

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
        chat_layout.addWidget(self.chat_history, 1)  # 分配更多空间给聊天历史

        # 右侧面板
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)

        # 创建选项卡窗口
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background: #232323;
                border-radius: 3px;
            }
            QTabBar::tab {
                background: #333333;
                color: #b0b0b0;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #4a90e2;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #444444;
            }
        """)

        # 思考过程选项卡
        self.thinking_widget = QWidget()
        thinking_layout = QVBoxLayout(self.thinking_widget)

        # 思考区域标题
        thinking_title = QLabel("AI 思考过程")
        thinking_title.setAlignment(Qt.AlignCenter)
        thinking_title.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 14px;")
        thinking_layout.addWidget(thinking_title)

        # 思考内容显示
        self.thinking_display = QTextEdit()
        self.thinking_display.setReadOnly(True)
        self.thinking_display.setFont(QFont("Arial", 10))
        self.thinking_display.setStyleSheet("""
            QTextEdit {
                background-color: #232323;
                color: #b0b0b0;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 5px;
            }
        """)
        self.thinking_display.setLineWrapMode(QTextEdit.WidgetWidth)
        thinking_layout.addWidget(self.thinking_display, 1)

        # 控制台选项卡
        self.console_widget = QWidget()
        console_layout = QVBoxLayout(self.console_widget)

        # 控制台标题区域
        console_header = QHBoxLayout()
        console_title = QLabel("系统日志")
        console_title.setStyleSheet("color: #4a90e2; font-weight: bold; font-size: 14px;")
        console_header.addWidget(console_title)

        # 添加清除按钮
        clear_console_btn = QPushButton("清除日志")
        clear_console_btn.setFixedWidth(80)
        clear_console_btn.clicked.connect(lambda: self.console_display.clear())
        console_header.addWidget(clear_console_btn)

        console_layout.addLayout(console_header)

        # 控制台显示
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)
        self.console_display.setFont(QFont("Courier New", 10))  # 使用等宽字体
        self.console_display.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #33ee33;  /* 绿色文本模拟终端 */
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                padding: 5px;
                font-family: 'Courier New', monospace;
            }
        """)
        self.console_display.setLineWrapMode(QTextEdit.WidgetWidth)
        console_layout.addWidget(self.console_display, 1)

        # 添加两个选项卡
        self.tabs.addTab(self.thinking_widget, "思考过程")
        self.tabs.addTab(self.console_widget, "系统日志")

        # 添加选项卡到右侧布局
        right_layout.addWidget(self.tabs)

        # 添加到分割器
        self.splitter.addWidget(self.chat_widget)
        self.splitter.addWidget(self.right_panel)  # 使用新的右侧面板

        # 设置初始分割比例 (3:1)
        self.splitter.setSizes([int(self.width() * 0.75), int(self.width() * 0.25)])

        # 添加分割器到主布局
        main_layout.addWidget(self.splitter, 1)  # 添加拉伸因子使其占据大部分空间

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

        # 将输入区域添加到主布局
        main_layout.addLayout(input_layout)

        # 设置主窗口的中央组件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 主窗口样式表
        self.set_background_styleSheet()

    def set_background_styleSheet(self):
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
            QCheckBox {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #4a90e2;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
            }
        """)

    def update_similarity_threshold_label(self):
        """更新相似度阈值标签显示"""
        value = self.similarity_threshold_slider.value() / 100
        self.similarity_threshold_label.setText(f"{value:.2f}")

    def log_to_console(self, message, message_type="info"):
        """
        向控制台面板添加日志信息

        参数:
            message (str): 日志信息
            message_type (str): 日志类型 - info, warning, error, success
        """
        timestamp = time.strftime("%H:%M:%S", time.localtime())

        # 根据消息类型设置颜色
        if message_type == "info":
            color = "#33ee33"  # 绿色
        elif message_type == "warning":
            color = "#ffcc00"  # 黄色
        elif message_type == "error":
            color = "#ff3333"  # 红色
        elif message_type == "success":
            color = "#33ccff"  # 蓝色
        else:
            color = "#ffffff"  # 白色

        # 构建HTML格式的日志消息
        log_entry = f'<span style="color: #aaaaaa;">[{timestamp}]</span> <span style="color: {color};">{message}</span><br>'

        # 添加到控制台显示
        self.console_display.append(log_entry)

        # 滚动到底部
        self.console_display.verticalScrollBar().setValue(
            self.console_display.verticalScrollBar().maximum())

        # 同时打印到标准输出（便于调试）
        print(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """在窗口关闭时处理线程终止"""
        print("正在关闭窗口，清理线程...")

        # 禁用UI，防止用户在关闭过程中进行交互
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # 停止思考线程
        if hasattr(self, 'thought_generator') and self.thought_generator:
            print("正在停止思考线程...")
            self.thought_generator._stop = True

            # 等待最多2秒让线程终止
            if not self.thought_generator.wait(2000):
                print("思考线程未能正常终止，强制继续关闭")

        # 停止聊天线程
        if hasattr(self, 'chat_thread') and self.chat_thread and self.chat_thread.isRunning():
            print("正在停止聊天线程...")
            self.chat_thread._stop = True

            # 等待最多2秒让线程终止
            if not self.chat_thread.wait(2000):
                print("聊天线程未能正常终止，强制继续关闭")

        # 恢复光标
        QApplication.restoreOverrideCursor()

        print("所有线程已停止，窗口即将关闭")
        event.accept()

    # 添加新的清除所有记忆方法
    def clear_all_memories(self):
        """清空所有存储的记忆"""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.memories = []
            if self.memory_manager.index is not None:
                # 保持索引维度，但清空内容
                dimension = self.memory_manager.dimension
                self.memory_manager.index = faiss.IndexFlatL2(dimension)
            self.memory_manager.save_memories()
            self.update_memory_count()
            self.chat_history.append('<div style="color:#808080;"><i>--- 所有记忆已清除 ---</i></div>')

    def update_embedding_model(self, model_name):
        """当嵌入模型改变时更新记忆管理器"""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.embedding_model = model_name
            self.chat_history.append(f'<div style="color:#808080;"><i>--- 嵌入模型已更改为: {model_name} ---</i></div>')

    def update_memory_count(self):
        """更新记忆数量显示"""
        try:
            if hasattr(self, 'memory_manager') and self.memory_manager:
                count = len(self.memory_manager.memories)
                self.memory_count.setText(f"记忆数量: {count}")

                # 如果记忆功能已启用，也更新状态文本
                if self.use_memory_cb.isChecked():
                    self.memory_status.setText(f"记忆状态: 已启用 ({count} 条)")
            else:
                self.memory_count.setText("记忆数量: 0")
        except Exception as e:
            print(f"更新记忆数量出错: {str(e)}")

    def toggle_memory_usage(self, state):
        """处理记忆使用选项的开关"""
        try:
            if state == Qt.Checked:  # 如果勾选了使用记忆
                self.log_to_console("正在启用记忆功能...", "info")

                # 初始化记忆管理器（如果尚未初始化）
                if not hasattr(self, 'memory_manager') or self.memory_manager is None:
                    self.memory_manager = OllamaMemoryManager(self.client)

                    # 初始化记忆检索器
                    from modules.memory_retriever import MemoryRetriever
                    self.memory_retriever = MemoryRetriever(self.memory_manager)

                    # 更新嵌入模型
                    current_embed_model = self.embed_model_combo.currentText()
                    self.memory_manager.embedding_model = current_embed_model
                    self.log_to_console(f"使用嵌入模型: {current_embed_model}", "info")

                    # 更新记忆数量显示
                    self.update_memory_count()

                    memory_count = len(self.memory_manager.memories) if hasattr(self.memory_manager, 'memories') else 0
                    if memory_count > 0:
                        self.chat_history.append(
                            f'<div style="color:#808080;"><i>--- 已加载 {memory_count} 条记忆 ---</i></div>')
                        self.log_to_console(f"成功加载 {memory_count} 条记忆", "success")

                        # 打印记忆列表和关键词
                        self.log_to_console("--- 记忆列表 ---", "info")

                        # 导入所需模块
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        import numpy as np
                        import re

                        # 使用TF-IDF提取关键词
                        def extract_keywords(text, top_n=5):
                            # 简单清洗文本
                            text = re.sub(r'[^\w\s]', '', text)
                            words = text.split()

                            # 如果文本太短，直接返回前几个词
                            if len(words) <= top_n:
                                return words

                            # 否则使用TF-IDF
                            try:
                                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                                tfidf_matrix = vectorizer.fit_transform([text])
                                feature_names = vectorizer.get_feature_names_out()

                                # 获取词语重要性分数
                                dense = tfidf_matrix.todense()
                                scores = np.asarray(dense)[0]

                                # 创建词-分数对并排序
                                word_scores = [(word, scores[idx]) for idx, word in enumerate(feature_names)]
                                word_scores.sort(key=lambda x: x[1], reverse=True)

                                # 返回前top_n个关键词
                                return [word for word, score in word_scores[:top_n]]
                            except:
                                # 如果TF-IDF失败，回退到简单方法
                                return words[:top_n]

                        # 遍历显示记忆及其关键词
                        for idx, memory in enumerate(self.memory_manager.memories):
                            memory_id = memory.get('id', f'mem_{idx}')
                            category = memory.get('category', '未分类')
                            thought = memory.get('thought', '')
                            created_at = memory.get('created_at', '未知时间')

                            # 提取关键词
                            keywords = extract_keywords(thought)
                            keywords_str = ", ".join(keywords)

                            # 准备摘要文本（太长的话截断）
                            summary = thought[:80] + "..." if len(thought) > 80 else thought

                            # 记录到控制台
                            log_entry = (f"记忆 #{idx + 1} [{memory_id}] - 类别: {category}, "
                                         f"时间: {created_at}\n"
                                         f"关键词: {keywords_str}\n"
                                         f"摘要: {summary}")
                            self.log_to_console(log_entry, "info")

                            # 每3条记忆后添加一个分隔符，提高可读性
                            if (idx + 1) % 3 == 0 and idx < len(self.memory_manager.memories) - 1:
                                self.log_to_console("---", "info")
                    else:
                        self.chat_history.append(
                            '<div style="color:#808080;"><i>--- 没有找到已存储的记忆 ---</i></div>')
                        self.log_to_console("没有找到已存储的记忆", "warning")

                self.memory_status.setText(f"记忆状态: 已启用 ({len(self.memory_manager.memories)} 条)")
                self.log_to_console("记忆功能已启用", "success")
            else:
                # 禁用记忆使用（但不删除记忆）
                self.memory_status.setText("记忆状态: 已禁用")
                self.chat_history.append('<div style="color:#808080;"><i>--- 记忆功能已禁用 ---</i></div>')
                self.log_to_console("记忆功能已禁用", "warning")
        except Exception as e:
            import traceback
            error_msg = f"切换记忆状态出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')
            self.log_to_console(f"记忆功能出错: {str(e)}", "error")
            # 出错时取消勾选
            self.use_memory_cb.setChecked(False)

    def show_memory_statistics(self):
        """显示记忆统计信息"""
        try:
            if not hasattr(self, 'memory_manager') or not self.memory_manager or not self.memory_manager.memories:
                self.log_to_console("没有可用的记忆进行统计", "warning")
                return

            memories = self.memory_manager.memories
            total_memories = len(memories)

            # 按类别统计
            categories = {}
            for memory in memories:
                cat = memory.get('category', '未分类')
                if cat in categories:
                    categories[cat] += 1
                else:
                    categories[cat] = 1

            # 按时间统计
            import datetime
            time_periods = {
                '今天': 0,
                '昨天': 0,
                '本周': 0,
                '本月': 0,
                '更早': 0
            }

            today = datetime.datetime.now().date()
            yesterday = today - datetime.timedelta(days=1)
            week_start = today - datetime.timedelta(days=today.weekday())
            month_start = datetime.datetime(today.year, today.month, 1).date()

            for memory in memories:
                if 'created_at' in memory:
                    try:
                        # 假设格式为 "YYYY-MM-DD HH:MM:SS"
                        time_str = memory['created_at']
                        memory_time = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").date()

                        if memory_time == today:
                            time_periods['今天'] += 1
                        elif memory_time == yesterday:
                            time_periods['昨天'] += 1
                        elif memory_time >= week_start:
                            time_periods['本周'] += 1
                        elif memory_time >= month_start:
                            time_periods['本月'] += 1
                        else:
                            time_periods['更早'] += 1
                    except:
                        time_periods['未知'] = time_periods.get('未知', 0) + 1

            # 记录统计结果
            self.log_to_console(f"--- 记忆统计 (共 {total_memories} 条) ---", "success")

            # 类别统计
            self.log_to_console("按类别统计:", "info")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_memories) * 100
                self.log_to_console(f"  - {category}: {count}条 ({percentage:.1f}%)", "info")

            # 时间统计
            self.log_to_console("按时间统计:", "info")
            for period, count in time_periods.items():
                if count > 0:
                    percentage = (count / total_memories) * 100
                    self.log_to_console(f"  - {period}: {count}条 ({percentage:.1f}%)", "info")

            self.log_to_console("--- 统计完成 ---", "success")

        except Exception as e:
            import traceback
            error_msg = f"生成记忆统计出错: {str(e)}\n{traceback.format_exc()}"
            self.log_to_

    def toggle_thinking(self):
        """切换思考状态"""
        try:
            # 获取当前勾选状态
            is_checked = self.thinking_toggle.isChecked()

            # 禁用按钮，防止用户重复点击
            self.thinking_toggle.setEnabled(False)

            if is_checked:
                # 启动思考
                self.log_to_console("正在启动思考进程...", "info")
                self.start_thinking()
            else:
                # 停止思考
                self.log_to_console("正在停止思考进程...", "info")
                self.stop_thinking()

        except Exception as e:
            import traceback
            error_msg = f"思考过程切换出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')
            # 出错时重置勾选状态
            self.thinking_toggle.setChecked(False)
        finally:
            # 重新启用按钮
            self.thinking_toggle.setEnabled(True)

    # In chat_window.py, modify the start_thinking method
    def start_thinking(self):
        """启动思考进程"""
        try:
            # 检查必要组件是否初始化
            if not hasattr(self, 'memory_manager') or self.memory_manager is None:
                print("初始化内存管理器...")

                # 用于总结的模型应该是聊天模型而不是嵌入模型
                chat_model = self.model_combo.currentText()  # 使用当前选择的聊天模型
                embed_model = self.embed_model_combo.currentText()  # 使用当前选择的嵌入模型

                self.log_to_console(f"使用嵌入模型: {embed_model}, 聊天模型: {chat_model}", "info")

                # 创建一个新的 OllamaMemoryManager 实例
                from modules.ollama_memory_manager import OllamaMemoryManager
                self.memory_manager = OllamaMemoryManager(
                    client=self.client,
                    memory_dir="memory",
                    embedding_model=embed_model,
                    chat_model=chat_model  # 添加聊天模型参数
                )

                self.log_to_console("初始化记忆管理器完成", "success")
            model = self.model_combo.currentText()

            # 创建思考生成器并传入记忆管理器
            self.thought_generator = ThoughtGenerator(self.client, model, self.memory_manager)

            # 连接新的信号
            self.thought_generator.thought_chunk_signal.connect(self.update_thinking_chunk)
            self.thought_generator.thought_complete_signal.connect(self.on_thought_complete)
            self.thought_generator.thinking_status.connect(self.update_thinking_status)

            self.log_to_console("启动思考线程...", "info")
            self.thought_generator.start()

            self.is_thinking = True
            self.memory_status.setText("记忆状态: 正在思考...")
            self.thinking_display.append(
                '<div style="color:#4a90e2;"><i>开始自我思考过程，基于记忆进行迭代发散...</i></div>')
            self.log_to_console("思考进程已成功启动", "success")
        except Exception as e:
            import traceback
            error_msg = f"启动思考过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')
            self.log_to_console(f"启动思考失败: {str(e)}", "error")
            # 出错时取消勾选
            self.thinking_toggle.setChecked(False)

    def on_thought_complete(self, thought, category):
        """当一个完整的思考生成完毕"""
        try:
            # 记录完整思考
            self.current_thinking = thought
            self.current_thinking_category = category

            # 更新分类信息
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            self.thinking_display.append(
                f'<div style="color:#c0c0c0;"><b>[{timestamp}] 分类: {category}</b></div>')

            # 清除当前思考显示引用
            if hasattr(self, 'current_thinking_display'):
                delattr(self, 'current_thinking_display')

            # 确保思考内容有意义
            if not thought or len(thought.strip()) < 50:
                self.log_to_console("生成的思考内容过短或为空，跳过记忆存储", "warning")
                return

            # 添加到记忆
            if hasattr(self, 'memory_manager') and self.memory_manager:
                # 通知用户正在进行总结 - 使用memory_status而不是thinking_status
                self.log_to_console("正在总结关键概念，此过程在后台运行...", "info")

                # 更新memory_status文本，而不是发送thinking_status信号
                self.memory_status.setText("记忆状态: 正在总结关键概念...")

                # 在思考显示区域也添加状态信息
                self.thinking_display.append(
                    '<div style="color:#808080;"><i>正在总结关键概念，此过程在后台运行...</i></div>')

                # 打印调试信息
                print(f"准备添加思考记忆，类别: {category}")
                print(f"思考内容长度: {len(thought)}")

                # 调用内存管理器添加记忆 - 这个过程现在不会卡住界面
                memory_id = self.memory_manager.add_memory(thought, category)

                if memory_id >= 0:
                    # 更新记忆计数
                    self.update_memory_count()
                    self.log_to_console(f"新增思考记忆，ID: {memory_id}, 类别: {category}", "success")

                    # 显示思考文本摘要
                    thought_summary = thought[:100] + "..." if len(thought) > 100 else thought
                    self.log_to_console(f"思考内容: {thought_summary}", "info")
                    self.log_to_console("关键概念正在后台生成，将自动更新...", "info")

                    # 更新记忆状态
                    self.memory_status.setText(f"记忆状态: 已添加记忆 #{memory_id}")
                else:
                    self.log_to_console("记忆存储失败", "error")
                    self.memory_status.setText("记忆状态: 记忆存储失败")
            else:
                self.log_to_console("记忆管理器未初始化，无法存储思考", "warning")

        except Exception as e:
            import traceback
            error_msg = f"处理完整思考内容出错: {str(e)}\n{traceback.format_exc()}"
            self.log_to_console(error_msg, "error")
            self.thinking_display.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')

    # 在 chat_window.py 中修改 update_thinking_chunk 方法
    def update_thinking_chunk(self, chunk):
        """处理思考流式输出的单个片段"""
        try:
            # 当开始新的思考时
            if not hasattr(self, 'current_thinking_display'):
                # 创建新的思考显示区块
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                self.thinking_display.append(
                    f'<div style="color:#c0c0c0;"><b>[{timestamp}] 思考中...</b></div>')

                # 创建一个块来累积这次思考的所有内容
                self.thinking_display.append('<div id="current_thinking" style="color:#e0e0e0;"></div>')

                # 获取当前光标位置
                cursor = self.thinking_display.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.current_thinking_display = cursor

                # 重置当前思考内容
                self.current_thinking = ""

            # 累积思考内容
            self.current_thinking += chunk

            # 更新整个思考区块的内容，而不是仅添加新片段
            # 找到最后一个"思考中..."后面的内容块
            cursor = self.thinking_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.PreviousBlock)

            # 替换整个块内容
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.insertHtml(f'<div id="current_thinking" style="color:#e0e0e0;">{self.current_thinking}</div>')

            # 滚动到底部
            self.thinking_display.verticalScrollBar().setValue(
                self.thinking_display.verticalScrollBar().maximum())

        except Exception as e:
            print(f"更新思考片段时出错: {str(e)}")

    def stop_thinking(self):
        """停止思考进程"""
        try:
            print("准备停止思考...")
            if hasattr(self, 'thought_generator') and self.thought_generator:
                self.thought_generator.stop()
                self.thought_generator = None

            self.is_thinking = False
            self.memory_status.setText("记忆状态: 空闲")
            self.thinking_display.append('<div style="color:#e07a7a;"><i>思考已停止</i></div>')
            print("思考已停止")
        except Exception as e:
            import traceback
            error_msg = f"停止思考过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')

    def update_thinking_status(self, status):
        """更新思考状态"""
        self.memory_status.setText(f"记忆状态: {status}")
        self.thinking_display.append(f'<div style="color:#808080;"><i>状态: {status}</i></div>')

    def on_thought_generated(self, thought, category):
        """处理生成的思考内容"""
        try:
            print(f"收到新思考: {category}")
            # 确保memory_manager已初始化
            if not hasattr(self, 'memory_manager') or self.memory_manager is None:
                self.memory_manager = OllamaMemoryManager(self.client)

            # 添加到记忆
            self.memory_manager.add_memory(thought, category)

            # 更新记忆计数
            self.update_memory_count()

            # 在思考显示区域显示
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            self.thinking_display.append(
                f'<div style="color:#c0c0c0;"><b>[{timestamp}] 分类: {category}</b></div>')
            self.thinking_display.append(
                f'<div style="color:#e0e0e0; margin-bottom:10px;">{thought}</div>')

            # 滚动到底部
            self.thinking_display.verticalScrollBar().setValue(
                self.thinking_display.verticalScrollBar().maximum())

        except Exception as e:
            import traceback
            error_msg = f"处理思考内容出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.thinking_display.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')

    def send_message(self):
        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            return

        was_thinking = False
        if self.is_thinking and hasattr(self, 'thought_generator') and self.thought_generator:
            was_thinking = True
            # 更新状态提示
            self.memory_status.setText("记忆状态: 正在停止思考...")
            self.log_to_console("正在停止思考线程以处理用户问题...", "warning")

            # 标记停止思考
            self.thought_generator._stop = True

            # 避免在后台立即重启思考
            self.thinking_toggle.setChecked(False)

            # 使UI暂时不响应，避免用户重复点击
            QApplication.setOverrideCursor(Qt.WaitCursor)

            try:
                # 等待线程完全停止，最多等3秒
                if not self.thought_generator.wait(3000):
                    # 如果3秒后线程仍未停止，记录警告
                    self.log_to_console("思考线程停止超时，强制继续...", "error")

                # 强制清理思考线程
                self.thought_generator = None
                self.is_thinking = False
                self.memory_status.setText("记忆状态: 已停止思考，正在回答问题...")
                self.thinking_display.append('<div style="color:#e07a7a;"><i>思考已强制停止，正在回答问题...</i></div>')
            finally:
                # 恢复正常光标
                QApplication.restoreOverrideCursor()

            # 确保事件循环处理已生成的事件
            QApplication.processEvents()

            # 短暂暂停确保资源释放
            time.sleep(0.2)

        # 更新服务器地址
        self.client.base_url = self.server_input.text().strip()

        # 添加用户消息到历史
        self.chat_history.append(f'<div style="color:#7ebeff;"><b>你:</b> {user_text}</div>')

        # 检查是否使用记忆
        if self.use_memory_cb.isChecked():
            self.log_to_console("正在使用记忆增强...", "info")

            # 确保记忆管理器已初始化
            if not hasattr(self, 'memory_manager') or self.memory_manager is None:
                self.memory_manager = OllamaMemoryManager(self.client)
                self.log_to_console("初始化记忆管理器", "info")
                # 初始化记忆检索器
                from modules.memory_retriever import MemoryRetriever
                self.memory_retriever = MemoryRetriever(self.memory_manager)

            # 如果记忆管理器存在且有记忆
            if hasattr(self, 'memory_manager') and self.memory_manager.memories:
                # 获取相关性阈值
                if hasattr(self, 'similarity_threshold_slider'):
                    threshold = self.similarity_threshold_slider.value() / 100
                else:
                    threshold = 0.65

                self.log_to_console(f"开始检索记忆，相关性阈值: {threshold:.2f}", "info")
                self.log_to_console(f"用户问题: \"{user_text}\"", "info")

                # 使用记忆增强提示
                if not hasattr(self, 'memory_retriever') or self.memory_retriever is None:
                    self.memory_retriever = MemoryRetriever(self.memory_manager)

                # 记录开始检索
                start_time = time.time()
                enhanced_query, memories = self.memory_retriever.enhance_prompt_with_memories(user_text,
                                                                                              top_k=3,
                                                                                              similarity_threshold=threshold)
                end_time = time.time()
                search_time = (end_time - start_time) * 1000  # 转换为毫秒

                # 如果找到了相关记忆，在聊天历史中显示
                if memories:
                    self.log_to_console(f"检索完成，用时: {search_time:.1f}毫秒，找到 {len(memories)} 条相关记忆",
                                        "success")

                    # 在日志中详细显示每条记忆的信息
                    for i, memory in enumerate(memories):
                        self.log_to_console(
                            f"记忆 #{i + 1}: 相关度 {memory['similarity']:.4f}, 类别: {memory['category']}", "info")
                        # 显示记忆文本摘要（太长的话只显示一部分）
                        memory_text = memory['thought']
                        if len(memory_text) > 100:
                            memory_text = memory_text[:97] + "..."
                        self.log_to_console(f"  内容: {memory_text}", "info")

                    similarities = [f"{memory['similarity']:.2f}" for memory in memories]
                    # 获取原始距离
                    raw_distances = [f"{memory.get('raw_distance', 0):.1f}" for memory in memories]
                    content_sims = [f"{memory.get('content_similarity', 0):.2f}" for memory in memories]

                    memory_info = (f"已找到 {len(memories)} 条相关记忆 "
                                   f"(相似度: {', '.join(similarities)}, "
                                   f"原始距离: {', '.join(raw_distances)}, "
                                   f"内容相似度: {', '.join(content_sims)})")

                    self.chat_history.append(f'<div style="color:#808080;"><i>--- {memory_info} ---</i></div>')
                else:
                    # 没有足够相关的记忆，使用原始查询
                    self.log_to_console(f"检索完成，用时: {search_time:.1f}毫秒，未找到足够相关的记忆", "warning")
                    self.chat_history.append(
                        '<div style="color:#808080;"><i>--- 未找到足够相关的记忆，使用原始查询 ---</i></div>')
                    self.messages.append({"role": "user", "content": user_text})
            else:
                # 没有记忆，提示用户
                self.log_to_console("没有可用的记忆，使用原始查询", "warning")
                self.chat_history.append('<div style="color:#808080;"><i>--- 没有可用的记忆，使用原始查询 ---</i></div>')
                self.messages.append({"role": "user", "content": user_text})
        else:
            # 不使用记忆，直接添加原始消息
            self.log_to_console("记忆功能未启用，使用原始查询", "info")
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
        try:
            self.chat_thread = ChatThread(
                self.client,
                model,
                self.messages.copy(),
                temperature,
                max_tokens
            )
            self.chat_thread.response_signal.connect(self.update_response)
            self.chat_thread.finished_signal.connect(lambda: self.complete_response(was_thinking))
            self.chat_thread.start()

            # 启用暂停按钮
            self.stop_button.setText("暂停")
            self.stop_button.setEnabled(True)
        except Exception as e:
            import traceback
            error_msg = f"启动聊天线程时出错: {str(e)}\n{traceback.format_exc()}"
            self.log_to_console(error_msg, "error")
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')

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
            if event.key() == Qt.Key_Return:
                if event.modifiers() & Qt.ShiftModifier:
                    return False
                else:
                    # 使用QTimer.singleShot延迟调用send_message，避免直接在事件处理中执行
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, self.send_message)
                    return True
        return super().eventFilter(obj, event)

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
    def complete_response(self, was_thinking=False):
        try:
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

            # 添加模型的回答到历史消息列表中
            self.messages.append({"role": "assistant", "content": response_for_history})

            # 如果原来在思考，且思考开关仍然打开，就恢复思考
            if was_thinking and self.thinking_toggle.isChecked():
                self.thinking_display.append('<div style="color:#4a90e2;"><i>问题回答完毕，恢复思考...</i></div>')
                self.start_thinking()
        except Exception as e:
            import traceback
            error_msg = f"处理回复时出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.chat_history.append(f'<div style="color:red;"><b>错误:</b> {error_msg}</div>')

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
