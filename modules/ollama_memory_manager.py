import math
import os
import json
import time
import numpy as np
import faiss


class OllamaMemoryManager:
    def __init__(self, client, memory_dir="memory", embedding_model="bge-m3"):
        # 创建记忆存储目录
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

        # 存储Ollama客户端
        self.client = client
        self.embedding_model = embedding_model

        # 初始化记忆和索引
        self.memories = []
        self.dimension = None  # 将根据第一个嵌入动态设置
        self.index = None

        # 加载已存在的记忆
        self.load_memories()

    def _initialize_index(self, dimension):
        """使用确定的维度初始化索引"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        print(f"索引已初始化，维度: {dimension}")

    def load_memories(self):
        """加载已存在的记忆"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)

                # 重建索引（如果有记忆）
                if self.memories:
                    # 从第一条记忆中获取维度
                    dimension = len(self.memories[0]["embedding"])
                    self._initialize_index(dimension)

                    # 添加所有嵌入到索引
                    embeddings = [np.array(m["embedding"], dtype=np.float32) for m in self.memories]
                    self.index.add(np.array(embeddings))

                    print(f"已加载 {len(self.memories)} 条记忆")
            except Exception as e:
                print(f"加载记忆失败: {e}")
                self.memories = []

    def save_memories(self):
        """保存记忆到文件"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            print(f"已保存 {len(self.memories)} 条记忆")
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def add_memory(self, thought, category="general"):
        """添加新的思考记忆，使用Ollama嵌入API"""
        try:
            # 使用Ollama嵌入API生成嵌入向量
            embedding_response = self.client.create_embedding(thought, self.embedding_model)
            if not embedding_response or not embedding_response["data"]:
                print("获取嵌入向量失败")
                return -1

            embedding = embedding_response["data"][0]["embedding"]

            # 如果索引尚未初始化，使用第一个嵌入的维度初始化
            if self.index is None:
                self._initialize_index(len(embedding))

            # 创建记忆条目
            memory = {
                "id": len(self.memories),
                "timestamp": time.time(),
                "thought": thought,
                "category": category,
                "embedding": embedding
            }

            # 添加到记忆列表
            self.memories.append(memory)

            # 更新索引
            self.index.add(np.array([embedding], dtype=np.float32))

            # 保存记忆
            self.save_memories()

            return memory["id"]
        except Exception as e:
            print(f"添加记忆失败: {e}")
            return -1

    def get_content_based_similarity(self, query, memory_text):
        """使用关键词匹配增强相似度评估"""
        # 简单的关键词提取和匹配
        import re
        from collections import Counter

        # 清理和分词函数
        def tokenize(text):
            # 转小写，去除特殊字符
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            # 分词
            return [word for word in text.split() if len(word) > 1]

        # 提取查询和记忆的词
        query_tokens = tokenize(query)
        memory_tokens = tokenize(memory_text)

        # 计算重合度
        if not query_tokens:
            return 0

        # 建立查询词频字典
        query_counter = Counter(query_tokens)

        # 计算匹配的查询词数量（考虑词频）
        matches = sum(min(query_counter.get(token, 0), 1) for token in memory_tokens)

        # 计算相似度 - 匹配词数除以查询词数
        return matches / len(query_tokens)

    def search_memories(self, query, top_k=5):
        """搜索相关记忆"""
        if not self.memories or self.index is None:
            return []

        try:
            # 使用Ollama嵌入API生成查询嵌入向量
            embedding_response = self.client.create_embedding(query, self.embedding_model)
            if not embedding_response or not embedding_response["data"]:
                print("获取查询嵌入向量失败")
                return []

            query_embedding = embedding_response["data"][0]["embedding"]

            # 检查查询嵌入维度是否与索引维度匹配
            if len(query_embedding) != self.dimension:
                print(f"查询嵌入维度 ({len(query_embedding)}) 与索引维度 ({self.dimension}) 不匹配")
                # 调整嵌入维度以匹配索引
                if len(query_embedding) > self.dimension:
                    query_embedding = query_embedding[:self.dimension]
                else:
                    query_embedding.extend([0] * (self.dimension - len(query_embedding)))

            # 搜索最相似的记忆
            distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32),
                                                   min(top_k, len(self.memories)))

            # 打印原始距离值用于调试
            print(f"原始FAISS距离值: {distances[0]}")

            # 返回结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.memories) and idx >= 0:
                    memory = self.memories[idx].copy()

                    # 获取原始距离
                    distance = distances[0][i]

                    # 使用余弦相似度的估计值 - 这需要记忆库中的嵌入已经被归一化
                    # 对于未归一化的嵌入，我们可以使用一个基于距离的相对相似度计算

                    # 使用min-max归一化，假设最大距离为1000（可以根据实际观察调整）
                    max_expected_distance = 1000.0
                    min_expected_distance = 0.0

                    # 距离归一化到[0,1]区间，然后反转（1-归一化距离）得到相似度
                    norm_distance = (distance - min_expected_distance) / (max_expected_distance - min_expected_distance)
                    norm_distance = max(0.0, min(1.0, norm_distance))  # 确保在[0,1]范围内
                    similarity = 1.0 - norm_distance

                    # 增强对比度 - 使高相似度更高，低相似度更低
                    # 可以使用sigmoid或幂函数来增强对比度
                    enhanced_similarity = similarity ** 0.5  # 开方会增强高相似度区域的对比度

                    memory["similarity"] = enhanced_similarity
                    memory["raw_distance"] = float(distance)  # 保留原始距离用于调试
                    memory.pop("embedding", None)  # 移除嵌入向量以减小数据量
                    results.append(memory)

            # 向结果添加基于内容的相似度增强
            for memory in results:
                # 计算内容相似度
                content_similarity = self.get_content_based_similarity(query, memory["thought"])
                # 原始相似度与内容相似度加权组合
                memory["similarity"] = 0.7 * memory["similarity"] + 0.3 * content_similarity
                memory["content_similarity"] = content_similarity  # 保存用于调试

            # 重新排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"搜索记忆失败: {e}\n{error_trace}")
            return []
