from PyQt5.QtCore import QThread, pyqtSignal


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
        self.running = True  # 控制是否继续生成

    def stop(self):
        self.running = False

    def run(self):
        stream_response = self.client.chat_completion(
            model=self.model,
            messages=self.messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        for chunk in stream_response:
            if not self.running:
                break  # 如果running为False，中断生成
            if chunk:
                content_delta = chunk["choices"][0]["delta"].get("content", "")
                if content_delta:
                    self.response_signal.emit(content_delta)

        self.finished_signal.emit()
