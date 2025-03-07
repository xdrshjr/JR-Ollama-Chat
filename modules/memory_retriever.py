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

    # 在 memory_retriever.py 中的 enhance_prompt_with_memories 方法中

    def enhance_prompt_with_memories(self, query, top_k=3, similarity_threshold=0.65):
        """
        用相关记忆增强用户查询，但只在相关性超过阈值时使用
        """
        # 获取相关记忆
        memories = self.retrieve_relevant_memories(query, top_k)

        # 调试信息
        if memories:
            print(f"检索到的记忆相似度: {[round(m['similarity'], 4) for m in memories]}")

        # 动态调整阈值 - 基于记忆相似度分布
        if memories:
            # 方法1: 使用最高相似度的一定比例作为阈值
            highest_similarity = memories[0]["similarity"]
            dynamic_threshold = max(0.3, highest_similarity * 0.8)  # 至少0.3，或最高相似度的80%

            # 方法2: 检测相似度断崖
            if len(memories) > 1:
                similarities = [m["similarity"] for m in memories]
                gaps = [similarities[i] - similarities[i + 1] for i in range(len(similarities) - 1)]
                if gaps and max(gaps) > 0.1:  # 如果有明显断崖
                    # 找到最大断崖的位置
                    cliff_index = gaps.index(max(gaps))
                    # 只使用断崖之前的记忆
                    memories = memories[:cliff_index + 1]
                    dynamic_threshold = 0  # 已经基于断崖筛选，不需要再用阈值
        else:
            dynamic_threshold = similarity_threshold

        # 打印动态阈值
        print(f"使用动态相似度阈值: {dynamic_threshold}")

        # 过滤出相关性高于阈值的记忆
        relevant_memories = [m for m in memories if m["similarity"] >= dynamic_threshold]

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