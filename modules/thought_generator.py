import random
import time

from PyQt5.QtCore import QThread, pyqtSignal


class ThoughtGenerator(QThread):
    thought_chunk_signal = pyqtSignal(str)  # 新增：用于流式输出思考片段
    thought_complete_signal = pyqtSignal(str, str)  # 完整思考内容和分类
    thinking_status = pyqtSignal(str)  # 用于UI状态更新

    def __init__(self, client, model, max_iterations=100):
        super().__init__()
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self._stop = False
        self.thinking_prompts = [
            "思考一个关于人工智能伦理的重要问题",
            "思考人类与技术的关系",
            # ... 其他提示保持不变
        ]
        self.categories = [
            "AI伦理", "人机关系", "未来展望", "哲学思考",
            # ... 其他分类保持不变
        ]
        self.current_thought = ""  # 当前正在生成的思考内容

    def stop(self):
        """安全地停止思考线程"""
        self._stop = True
        self.wait(1000)  # 等待最多1秒钟

    def run(self):
        """开始自主思考过程"""
        iteration = 0

        while not self._stop and iteration < self.max_iterations:
            # 随机选择一个思考提示和分类
            idx = random.randint(0, len(self.thinking_prompts) - 1)
            prompt = self.thinking_prompts[idx]
            category = self.categories[idx]

            self.thinking_status.emit(f"正在思考: {prompt}...")
            self.current_thought = ""  # 重置当前思考

            try:
                # 构建自我思考的消息
                messages = [
                    {"role": "system",
                     "content": "你是一个深度思考者，请对给定的主题进行深入、原创的思考。产生见解时避免陈词滥调，尝试提出新的角度。"},
                    {"role": "user", "content": f"{prompt}。请进行深入、系统化的思考，提出原创见解。"}
                ]

                # 使用流式响应进行思考
                stream_response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=0.8
                )

                # 处理流式响应
                for chunk in stream_response:
                    if self._stop:
                        break
                    if chunk:
                        content_delta = chunk["choices"][0]["delta"].get("content", "")
                        if content_delta:
                            self.current_thought += content_delta
                            self.thought_chunk_signal.emit(content_delta)  # 发送每个片段

                # 思考完成，发送完整内容
                if not self._stop:
                    self.thought_complete_signal.emit(self.current_thought, category)

                # 暂停一段时间，避免过快思考
                time.sleep(random.uniform(2, 5))

                iteration += 1

            except Exception as e:
                self.thinking_status.emit(f"思考过程中出错: {str(e)}")
                time.sleep(5)  # 出错后暂停

        self.thinking_status.emit("思考已完成")
