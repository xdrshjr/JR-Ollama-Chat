import math
import os
import json
import time
import numpy as np
import faiss


class OllamaMemoryManager:
    def __init__(self, client, memory_dir="memory", embedding_model="bge-m3", chat_model=None):
        # 创建记忆存储目录
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)

        # 存储Ollama客户端
        self.client = client
        self.embedding_model = embedding_model
        self.chat_model = chat_model if chat_model else embedding_model  # 如果未指定，默认使用嵌入模型

        # 初始化记忆和索引
        self.memories = []
        self.dimension = None  # 将根据第一个嵌入动态设置
        self.index = None

        # 加载已存在的记忆
        self.load_memories()

        print(f"初始化记忆管理器: 嵌入模型={embedding_model}, 聊天模型={self.chat_model}")

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

    # In ollama_memory_manager.py
    # 修改 OllamaMemoryManager 中的 add_memory 方法，确保记忆被正确存储

    def add_memory(self, thought, category="general"):
        """添加新的思考记忆，使用Ollama嵌入API，并生成关键句子总结"""
        try:
            # 检查思考内容是否为空
            if not thought or thought.strip() == "":
                print("思考内容为空，跳过存储")
                return -1

            # 提取关键句子 - 首先尝试使用LLM进行总结
            key_sentence = ""
            try:
                # 使用LLM生成关键句子总结
                summary_prompt = f"请用一句简洁的话总结以下思考的核心概念(30字以内) \n\n{thought}\n\n总结："

                # 调用API生成总结 - 使用聊天模型而不是嵌入模型
                print(f"调用聊天模型 {self.chat_model} 生成关键句子总结")
                summary_response = self.client.chat_completion(
                    model=self.chat_model,  # 使用专门的聊天模型
                    messages=[{"role": "user", "content": summary_prompt}],
                    stream=False,
                    temperature=0.3,
                    max_tokens=50
                )

                # 详细记录响应结构以便调试
                print(f"总结响应类型: {type(summary_response)}")
                if summary_response:
                    print(
                        f"总结响应键: {summary_response.keys() if isinstance(summary_response, dict) else 'Not a dict'}")

                # 检查响应是否有效并提取内容
                if summary_response and isinstance(summary_response, dict):
                    if "choices" in summary_response and summary_response["choices"]:
                        content = summary_response["choices"][0]["message"]["content"].strip()
                    # Ollama API可能使用不同的响应格式
                    elif "message" in summary_response and "content" in summary_response["message"]:
                        content = summary_response["message"]["content"].strip()
                    else:
                        print("无法从响应中提取关键句子")
                        print(f"响应内容: {summary_response}")
                        content = ""

                    # 提取</think>之后的内容
                    import re
                    think_match = re.search(r'</think>(.*?)$', content, re.DOTALL)
                    if think_match:
                        key_sentence = think_match.group(1).strip()
                        print(f"从</think>之后提取的核心概念: {key_sentence}")
                    else:
                        # 如果没有</think>标签，就使用完整内容
                        key_sentence = content
                        print(f"未找到</think>标签，使用完整内容: {key_sentence}")
                else:
                    print("LLM总结返回无效响应")
                    print(f"响应内容: {summary_response}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"使用LLM生成关键句子失败: {e}\n{error_trace}")

            # 如果LLM总结失败，回退到简单的提取方法
            if not key_sentence:
                print("使用备选方法提取关键句子")
                # 简单提取第一句话，最多100个字符
                sentences = thought.split('。')
                if sentences:
                    key_sentence = sentences[0].strip()[:100]
                    if len(key_sentence) == 100:
                        key_sentence += "..."
                else:
                    # 如果无法分句，直接截取开头
                    key_sentence = thought[:100].strip()
                    if len(thought) > 100:
                        key_sentence += "..."

            print(f"最终使用的关键句子: {key_sentence}")

            # 为关键句子生成嵌入向量
            embedding_response = self.client.create_embedding(key_sentence, self.embedding_model)
            if not embedding_response or not embedding_response["data"]:
                print("获取嵌入向量失败")
                return -1

            embedding = embedding_response["data"][0]["embedding"]

            # 如果索引尚未初始化，使用第一个嵌入的维度初始化
            if self.index is None:
                self._initialize_index(len(embedding))

            # 创建记忆条目
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            memory = {
                "id": len(self.memories),
                "timestamp": time.time(),
                "created_at": current_time,
                "thought": thought,
                "key_sentence": key_sentence,
                "category": category,
                "embedding": embedding
            }

            # 添加到记忆列表
            self.memories.append(memory)

            # 更新索引
            self.index.add(np.array([embedding], dtype=np.float32))

            # 保存记忆
            self.save_memories()

            print(f"成功添加记忆 #{memory['id']}, 关键句子: {key_sentence}")
            return memory["id"]
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"添加记忆失败: {e}\n{error_trace}")
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

    # In ollama_memory_manager.py
    def search_memories(self, query, top_k=5):
        """搜索相关记忆，基于关键句子的嵌入向量"""
        if not self.memories or self.index is None:
            return []

        try:
            # 调试日志
            print(f"搜索记忆: {query}")
            print(f"当前记忆数量: {len(self.memories)}")

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
            print(f"对应索引: {indices[0]}")

            # 返回结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.memories) and idx >= 0:
                    memory = self.memories[idx].copy()

                    # 确保包含关键句子
                    if "key_sentence" not in memory:
                        memory["key_sentence"] = "未生成关键句子"

                    # 获取原始距离
                    distance = distances[0][i]

                    # 计算相似度
                    max_expected_distance = 2.0  # 调整期望的最大距离
                    norm_distance = min(distance / max_expected_distance, 1.0)  # 归一化到[0,1]
                    similarity = 1.0 - norm_distance  # 转换为相似度

                    memory["similarity"] = similarity
                    memory["raw_distance"] = float(distance)
                    memory.pop("embedding", None)  # 移除嵌入向量以减小数据量
                    results.append(memory)

            # 向结果添加基于内容的相似度增强
            for memory in results:
                # 基于关键句子计算内容相似度
                key_sentence = memory.get("key_sentence", "")
                content_similarity = self.get_content_based_similarity(query, key_sentence) if key_sentence else 0

                # 结合向量相似度和内容相似度
                memory["similarity"] = 0.6 * memory["similarity"] + 0.4 * content_similarity
                memory["content_similarity"] = content_similarity

            # 重新排序
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # 调试输出搜索结果
            for i, res in enumerate(results):
                print(f"匹配项 #{i + 1}: 相似度={res['similarity']:.4f}, 关键句子: {res.get('key_sentence', 'N/A')}")

            return results
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"搜索记忆失败: {e}\n{error_trace}")
            return []
