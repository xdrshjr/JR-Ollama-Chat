import random
import time

from PyQt5.QtCore import QThread, pyqtSignal


# In thought_generator.py
# 完全重写 ThoughtGenerator 类以避免循环并实现真正的迭代思考

class ThoughtGenerator(QThread):
    thought_chunk_signal = pyqtSignal(str)
    thought_complete_signal = pyqtSignal(str, str)
    thinking_status = pyqtSignal(str)

    def __init__(self, client, model, memory_manager=None, max_iterations=100):
        super().__init__()
        self.client = client
        self.model = model
        self.memory_manager = memory_manager
        self.max_iterations = max_iterations
        self._stop = False
        self.current_thought = ""

        # 初始思考主题集，当没有前一个记忆时使用
        self.initial_topics = [
            "人工智能与人类创造力的关系",
            "技术进步对社会结构的影响",
            "数据隐私与个人自由的平衡",
            "人机协作如何改变工作方式",
            "大语言模型的认知局限性",
            "AI伦理决策的哲学基础",
            "技术如何影响人类记忆和认知",
            "数字世界中的身份与自我认知",
            "机器学习系统中的偏见问题",
            "人工智能时代的教育转型"
        ]

        # 思考类别
        self.categories = [
            "AI伦理", "人机关系", "未来展望", "哲学思考",
            "认知科学", "社会影响", "技术前沿", "创新思考",
            "数据伦理", "隐私安全", "语言模型", "人类意识"
        ]

        # 记录最后一次思考的时间，避免太频繁
        self.last_thought_time = 0

        # 记录已处理过的记忆ID，避免重复处理
        self.processed_memory_ids = set()

    def get_thinking_seed(self):
        """获取思考种子，可能来自最新记忆或初始主题"""
        # 检查是否有记忆管理器和记忆
        if not self.memory_manager or not self.memory_manager.memories:
            # 如果没有记忆，使用随机初始主题
            topic = random.choice(self.initial_topics)
            category = random.choice(self.categories)
            return topic, category, None

        # 获取最新的记忆
        latest_memories = sorted(self.memory_manager.memories, key=lambda m: m.get("timestamp", 0), reverse=True)

        # 检查是否有未处理过的记忆
        unprocessed_memories = [m for m in latest_memories if m["id"] not in self.processed_memory_ids]

        if not unprocessed_memories:
            # 如果所有记忆都已处理过，使用最新记忆再次启发，但用不同角度
            if latest_memories:
                latest_memory = latest_memories[0]
                key_sentence = latest_memory.get("key_sentence", "")
                category = random.choice(self.categories)  # 换一个类别

                # 将最新记忆ID标记为已处理
                self.processed_memory_ids.add(latest_memory["id"])

                if key_sentence:
                    # 创建一个基于最新记忆但角度不同的思考种子
                    new_angle_prompt = f"从新角度拓展这个概念: {key_sentence}"
                    return new_angle_prompt, category, latest_memory["id"]

            # 如果没有记忆或出现问题，回退到随机主题
            topic = random.choice(self.initial_topics)
            category = random.choice(self.categories)
            return topic, category, None

        # 使用最新未处理的记忆
        latest_memory = unprocessed_memories[0]
        key_sentence = latest_memory.get("key_sentence", "")
        category = latest_memory.get("category", random.choice(self.categories))

        # 将最新记忆ID标记为已处理
        self.processed_memory_ids.add(latest_memory["id"])

        if key_sentence:
            # 创建一个基于最新记忆的思考种子
            seed_prompt = f"深入探索并拓展这个概念: {key_sentence}"
            return seed_prompt, category, latest_memory["id"]
        else:
            # 如果没有关键句子，使用随机主题
            topic = random.choice(self.initial_topics)
            return topic, category, latest_memory["id"]

    def run(self):
        """执行迭代思考过程"""
        iteration = 0

        while not self._stop and iteration < self.max_iterations:
            # 检查上次思考时间，确保有足够间隔
            current_time = time.time()
            if current_time - self.last_thought_time < 3:  # 至少间隔3秒
                time.sleep(1)
                continue

            # 获取思考种子
            prompt, category, memory_id = self.get_thinking_seed()

            # 更新状态
            self.thinking_status.emit(f"正在思考: {prompt}...")
            self.current_thought = ""

            try:
                # 构建思考提示
                system_prompt = "你是一个深度思考者，请对给定的主题进行深入、原创的思考。思考过程中要拓展概念边界，提出新的联系，避免陈词滥调。"

                if memory_id is not None:
                    user_prompt = f"{prompt}\n\n请进行深入、系统化的思考，基于这个核心概念发展出新的思路和见解。不要重复原有概念，而是向新方向拓展。"
                else:
                    user_prompt = f"{prompt}\n\n请进行深入、系统化的思考，提出原创见解。"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # 进行思考
                stream_response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=0.8,
                    max_tokens=1000
                )

                # 处理流式输出
                for chunk in stream_response:
                    if self._stop:
                        break
                    if chunk:
                        content_delta = chunk["choices"][0]["delta"].get("content", "")
                        if content_delta:
                            self.current_thought += content_delta
                            self.thought_chunk_signal.emit(content_delta)

                # 思考完成后，确保内容非空
                if not self._stop and self.current_thought and len(self.current_thought.strip()) > 50:
                    # 发送完整思考结果
                    self.thought_complete_signal.emit(self.current_thought, category)
                    # 更新最后思考时间
                    self.last_thought_time = time.time()
                    # 增加迭代计数
                    iteration += 1
                    # 随机暂停一段时间
                    pause_time = random.uniform(15, 30)
                    self.thinking_status.emit(f"思考已完成，休息 {pause_time:.1f} 秒...")
                    time.sleep(pause_time)
                else:
                    # 如果思考结果为空或过短，暂停短时间后重试
                    self.thinking_status.emit("思考结果不理想，准备重新思考...")
                    time.sleep(3)

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                self.thinking_status.emit(f"思考过程中出错: {str(e)}")
                print(f"思考错误: {e}\n{error_trace}")
                time.sleep(5)  # 出错后暂停

        self.thinking_status.emit("思考循环已完成")
