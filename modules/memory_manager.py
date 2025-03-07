import os
import json
import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class MemoryManager:
    def __init__(self, memory_dir="memory"):
        # 创建记忆存储目录
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

        # 加载嵌入模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.model.get_sentence_embedding_dimension()

        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(self.vector_size)
        self.memories = []

        # 加载已存在的记忆
        self.load_memories()

    def load_memories(self):
        """加载已存在的记忆"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)

                # 重建索引
                if self.memories:
                    embeddings = [np.array(m["embedding"], dtype=np.float32) for m in self.memories]
                    self.index = faiss.IndexFlatL2(self.vector_size)
                    self.index.add(np.array(embeddings))
            except Exception as e:
                print(f"加载记忆失败: {e}")
                self.memories = []
                self.index = faiss.IndexFlatL2(self.vector_size)

    def save_memories(self):
        """保存记忆到文件"""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def add_memory(self, thought, category="general"):
        """添加新的思考记忆"""
        # 计算嵌入向量
        embedding = self.model.encode(thought)

        # 创建记忆条目
        memory = {
            "id": len(self.memories),
            "timestamp": time.time(),
            "thought": thought,
            "category": category,
            "embedding": embedding.tolist()
        }

        # 添加到记忆列表
        self.memories.append(memory)

        # 更新索引
        self.index.add(np.array([embedding]))

        # 保存记忆
        self.save_memories()

        return memory["id"]

    def search_memories(self, query, top_k=5):
        """搜索相关记忆"""
        if not self.memories:
            return []

        # 计算查询的嵌入向量
        query_embedding = self.model.encode(query)

        # 搜索最相似的记忆
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories) and idx >= 0:
                memory = self.memories[idx].copy()
                memory["similarity"] = float(1 - distances[0][i] / 10)  # 归一化相似度
                memory.pop("embedding", None)  # 移除嵌入向量
                results.append(memory)

        return results