import random
import time
from PyQt5.QtCore import QThread, pyqtSignal


class ThoughtGenerator(QThread):
    thought_generated = pyqtSignal(str, str)  # 信号: 思考内容, 分类
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
            "想象未来世界会是什么样子",
            "思考一个哲学概念并深入探讨",
            "思考当前科技的优缺点",
            "思考数据隐私与安全的挑战",
            "思考学习和知识获取的过程",
            "分析一个社会现象的成因和影响",
            "思考如何解决一个常见的技术难题",
            "思考创造力和创新的源泉"
        ]

        self.categories = [
            "AI伦理", "人机关系", "未来展望", "哲学思考",
            "技术评估", "数据安全", "学习理论", "社会分析",
            "技术难题", "创新思维"
        ]

    def stop(self):
        """停止思考过程"""
        self._stop = True

    def run(self):
        """开始自主思考过程"""
        iteration = 0

        while not self._stop and iteration < self.max_iterations:
            # 随机选择一个思考提示和分类
            idx = random.randint(0, len(self.thinking_prompts) - 1)
            prompt = self.thinking_prompts[idx]
            category = self.categories[idx]

            self.thinking_status.emit(f"正在思考: {prompt}...")

            try:
                # 构建自我思考的消息
                messages = [
                    {"role": "system",
                     "content": "你是一个深度思考者，请对给定的主题进行深入、原创的思考。产生见解时避免陈词滥调，尝试提出新的角度。"},
                    {"role": "user", "content": f"{prompt}。请进行深入、系统化的思考，提出原创见解。"}
                ]

                # 调用AI模型进行思考
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    temperature=0.8
                )

                thought = response["choices"][0]["message"]["content"]

                # 发送生成的思考内容
                self.thought_generated.emit(thought, category)

                # 暂停一段时间，避免过快思考
                time.sleep(random.uniform(2, 5))

                iteration += 1

            except Exception as e:
                self.thinking_status.emit(f"思考过程中出错: {str(e)}")
                time.sleep(5)  # 出错后暂停

        self.thinking_status.emit("思考已完成")
