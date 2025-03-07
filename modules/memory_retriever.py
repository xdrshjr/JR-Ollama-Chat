class MemoryRetriever:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        # 删除这里错误的 self.initUI() 调用

    def retrieve_relevant_memories(self, query, top_k=3):
        """检索与查询相关的记忆"""
        return self.memory_manager.search_memories(query, top_k)

    def format_memories_for_context(self, memories):
        """将记忆格式化为上下文信息"""
        # 先按相似度排序
        memories.sort(key=lambda x: x["similarity"], reverse=True)

        formatted_entries = []
        for i, memory in enumerate(memories):
            # 获取子序列分析结果
            subsequence_analysis = memory.get("subsequence_analysis", {})
            top_subsequences = subsequence_analysis.get("top_subsequences", [])

            # 如果有高相似度的子序列，只展示最相关的部分
            if top_subsequences and top_subsequences[0]["similarity"] >= 0.5:
                # 构建突出显示相关子序列的内容
                relevant_text = "相关段落:\n"
                for j, subseq in enumerate(top_subsequences):
                    if subseq["similarity"] >= 0.3:  # 只包含足够相关的子序列
                        relevant_text += f"- {subseq['text']}\n"

                # 添加记忆条目，突出显示相关部分
                formatted_entries.append(
                    f"记忆 {i + 1}（类别: {memory.get('category', '未分类')}）:\n"
                    f"{relevant_text}\n"
                    f"完整思考:\n{memory.get('thought', '')}"
                )
            else:
                # 没有特别相关的子序列，展示完整内容
                formatted_entries.append(
                    f"记忆 {i + 1}（类别: {memory.get('category', '未分类')}）:\n"
                    f"{memory.get('thought', '')}"
                )

        # 合并所有格式化的条目
        return "以下是我过去的相关思考:\n\n" + "\n\n".join(formatted_entries)

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
        # 使用子序列分析的最大相似度和整体相似度的组合来判断
        relevant_memories = []
        for memory in memories:
            # 检查子序列分析结果
            if "subsequence_analysis" in memory and memory["subsequence_analysis"][
                "max_similarity"] >= similarity_threshold:
                # 有高相似度的子序列，视为相关
                memory["relevance_reason"] = "子序列相似度高"
                relevant_memories.append(memory)
            # 备选：整体相似度也很高
            elif memory["similarity"] >= similarity_threshold:
                memory["relevance_reason"] = "整体相似度高"
                relevant_memories.append(memory)

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