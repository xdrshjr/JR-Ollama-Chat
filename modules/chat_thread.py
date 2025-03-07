from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot


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
        # 存储当前正在生成的思考
        self.current_thinking = ""
        self.current_thinking_category = ""

    # 在 thought_generator.py 中添加
    def stop(self):
        """停止思考进程"""
        self._stop = True
        # 等待线程实际终止的辅助方法
        self.wait(1000)  # 最多等待1秒

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
