class MemoryRetriever:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        # 删除这里错误的 self.initUI() 调用

    def retrieve_relevant_memories(self, query, top_k=3):
        """检索与查询相关的记忆"""
        return self.memory_manager.search_memories(query, top_k)

    def format_memories_for_context(self, memories):
        """将记忆格式化为上下文文本"""
        if not memories:
            return ""

        memories_text = "以下是我过去思考过的相关内容:\n\n"

        for i, memory in enumerate(memories):
            memories_text += f"{i + 1}. {memory['thought']}\n"
            memories_text += f"   [类别: {memory['category']}, 相关度: {memory['similarity']:.2f}]\n\n"

        return memories_text

    def enhance_prompt_with_memories(self, query, top_k=3, similarity_threshold=0.6):
        """
        用相关记忆增强用户查询，但只在相关性超过阈值时使用

        参数:
            query (str): 用户的原始查询
            top_k (int): 检索的最大记忆数量
            similarity_threshold (float): 相关性阈值，0-1之间，只有超过这个阈值的记忆才会被使用

        返回:
            (str, list): 增强后的查询和使用的相关记忆列表
        """
        # 获取相关记忆
        memories = self.retrieve_relevant_memories(query, top_k)

        # 过滤出相关性高于阈值的记忆
        relevant_memories = [memory for memory in memories if memory['similarity'] >= similarity_threshold]

        # 检查是否有满足条件的记忆
        if relevant_memories:
            # 将符合相关性阈值的记忆格式化为上下文
            memories_context = self.format_memories_for_context(relevant_memories)
            # 构建增强查询
            enhanced_query = f"{query}\n\n{memories_context}\n请基于以上内容和我过去的思考回答问题。如果这些思考与当前问题无关，请忽略它们直接回答问题。"
            return enhanced_query, relevant_memories
        else:
            # 没有相关记忆，返回原始查询
            return query, []