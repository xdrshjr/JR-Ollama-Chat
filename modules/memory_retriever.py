class MemoryRetriever:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        # 删除这里错误的 self.initUI() 调用

    def retrieve_relevant_memories(self, query, top_k=3):
        """检索与查询相关的记忆"""
        return self.memory_manager.search_memories(query, top_k)

    # In memory_retriever.py, update format_memories_for_context method
    def format_memories_for_context(self, memories):
        """将记忆格式化为上下文信息，突出显示关键句子"""
        # 按相似度排序
        memories.sort(key=lambda x: x["similarity"], reverse=True)

        formatted_entries = []
        for i, memory in enumerate(memories):
            # 获取关键句子
            key_sentence = memory.get("key_sentence", "")

            # 构建显示内容
            if key_sentence:
                formatted_entries.append(
                    f"记忆 {i + 1}（类别: {memory.get('category', '未分类')}）:\n"
                    f"核心概念: {key_sentence}\n"
                    f"完整思考:\n{memory.get('thought', '')}"
                )
            else:
                # 没有关键句子，展示完整内容
                formatted_entries.append(
                    f"记忆 {i + 1}（类别: {memory.get('category', '未分类')}）:\n"
                    f"{memory.get('thought', '')}"
                )

        # 合并所有格式化的条目
        return "以下是我过去的相关思考:\n\n" + "\n\n".join(formatted_entries)

    # 在 memory_retriever.py 中的 enhance_prompt_with_memories 方法中

    def enhance_prompt_with_memories(self, query, top_k=3, similarity_threshold=0.65):
        """
        用相关记忆增强用户查询，但只选择最相关的一条记忆
        """
        # 获取相关记忆
        memories = self.retrieve_relevant_memories(query, top_k)

        # 调试信息
        if memories:
            print(f"检索到的记忆相似度: {[round(m['similarity'], 4) for m in memories]}")

        # 检查是否有记忆，并且最高相似度的记忆超过阈值
        if memories and memories[0]["similarity"] >= similarity_threshold:
            # 选择最相关的一条记忆
            best_memory = memories[0]
            print(f"使用最高相似度记忆: {round(best_memory['similarity'], 4)}")

            # 将最相关的记忆格式化为上下文
            memory_context = self.format_memories_for_context([best_memory])

            # 构建增强查询
            enhanced_query = f"[Query]\n{query}\n\n[Memory]{memory_context}\n请基于以上内容和我过去的思考回答问题。如果这些思考与当前问题无关，请忽略它们直接回答问题。"
            return enhanced_query, [best_memory]
        else:
            # 没有相关记忆，或相似度不够高，返回原始查询
            if memories:
                print(f"最高相似度 {round(memories[0]['similarity'], 4)} 低于阈值 {similarity_threshold}，不使用记忆")
            else:
                print("未检索到相关记忆")
            return query, []