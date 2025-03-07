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

    def enhance_prompt_with_memories(self, query, top_k=3):
        """用相关记忆增强用户查询"""
        memories = self.retrieve_relevant_memories(query, top_k)
        memories_context = self.format_memories_for_context(memories)

        if memories_context:
            enhanced_query = f"{query}\n\n{memories_context}\n请基于以上内容和我过去的思考回答问题。"
            return enhanced_query, memories

        return query, []