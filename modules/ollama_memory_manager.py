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

                    # 查看我们使用的是哪种FAISS索引类型
                    if hasattr(self, 'index_type') and self.index_type == 'cosine':
                        # 对于余弦相似度：距离 = 1 - 相似度，所以相似度 = 1 - 距离
                        similarity = 1.0 - distance
                    elif hasattr(self, 'index_type') and self.index_type == 'ip':
                        # 对于内积，较大值表示更相似，但需要归一化
                        # 假设最大可能内积为理论最大值1.0
                        similarity = max(0.0, min(1.0, distance))
                    else:
                        # 对于欧几里得距离 (L2)，使用更合适的映射函数
                        scaling_factor = 0.01  # 可以根据实际距离分布调整
                        similarity = float(math.exp(-distance * scaling_factor)) * 100

                    # 确保相似度在有效范围内
                    similarity = max(0.0, min(1.0, similarity))

                    # 设置相似度，并保留原始距离以供调试
                    memory["similarity"] = similarity
                    memory["raw_distance"] = float(distance)
                    memory.pop("embedding", None)  # 移除嵌入向量以减小数据量
                    results.append(memory)

            # 按相似度从高到低排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"搜索记忆失败: {e}\n{error_trace}")
            return []