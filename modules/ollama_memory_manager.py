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

    def add_memory(self, thought, category="general"):
        """添加新的思考记忆，使用Ollama嵌入API，并生成关键句子总结"""
        try:
            # 检查思考内容是否为空
            if not thought or thought.strip() == "":
                print("思考内容为空，跳过存储")
                return -1

            # 先创建一个临时记忆，使用思考的前部分作为临时关键句子
            temp_key_sentence = thought[:100].strip() + "..." if len(thought) > 100 else thought

            # 通知用户正在处理关键句子
            print("正在总结记忆关键概念，请稍候...")

            # 计算嵌入向量（使用临时关键句子）
            embedding_response = self.client.create_embedding(temp_key_sentence, self.embedding_model)
            if not embedding_response or not embedding_response["data"]:
                print("获取嵌入向量失败")
                return -1

            embedding = embedding_response["data"][0]["embedding"]

            # 如果索引尚未初始化，使用第一个嵌入的维度初始化
            if self.index is None:
                self._initialize_index(len(embedding))

            # 创建临时记忆条目，先使用临时关键句子
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            memory_id = len(self.memories)
            memory = {
                "id": memory_id,
                "timestamp": time.time(),
                "created_at": current_time,
                "thought": thought,
                "key_sentence": temp_key_sentence,  # 临时关键句子
                "category": category,
                "embedding": embedding,
                "is_summarized": False  # 标记尚未完成总结
            }

            # 添加到记忆列表
            self.memories.append(memory)

            # 更新索引
            self.index.add(np.array([embedding], dtype=np.float32))

            # 保存记忆（临时版本）
            self.save_memories()

            # 启动一个单独的线程来处理关键句子总结
            import threading
            summarize_thread = threading.Thread(
                target=self._summarize_memory_async,
                args=(memory_id, thought, category)
            )
            summarize_thread.daemon = True  # 设为守护线程，防止程序退出时阻塞
            summarize_thread.start()

            print(f"成功添加记忆 #{memory_id}，关键句子总结正在后台处理中...")
            return memory_id
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"添加记忆失败: {e}\n{error_trace}")
            return -1

    def _summarize_memory_async(self, memory_id, thought, category):
        """异步处理记忆总结"""
        try:
            if memory_id >= len(self.memories):
                print(f"错误：找不到ID为{memory_id}的记忆")
                return

            print(f"开始为记忆 #{memory_id} 生成关键句子总结...")

            # 使用LLM生成关键句子总结
            summary_prompt = f"请用一句简洁的话总结以下思考的核心概念(30字以内) \n\n{thought}\n\n总结："

            # 设置超时（避免无限等待）
            import threading
            import time

            result = {"key_sentence": "", "success": False, "timeout": False}

            def call_api():
                try:
                    # 调用API生成总结 - 使用聊天模型而不是嵌入模型
                    summary_response = self.client.chat_completion(
                        model=self.chat_model,
                        messages=[{"role": "user", "content": summary_prompt}],
                        stream=False,
                        temperature=0.3,
                        max_tokens=50
                    )

                    if summary_response and isinstance(summary_response, dict):
                        if "choices" in summary_response and summary_response["choices"]:
                            content = summary_response["choices"][0]["message"]["content"].strip()
                        elif "message" in summary_response and "content" in summary_response["message"]:
                            content = summary_response["message"]["content"].strip()
                        else:
                            content = ""

                        # 提取</think>之后的内容
                        import re
                        think_match = re.search(r'</think>(.*?)$', content, re.DOTALL)
                        if think_match:
                            key_sentence = think_match.group(1).strip()
                        else:
                            key_sentence = content

                        result["key_sentence"] = key_sentence
                        result["success"] = True
                except Exception as e:
                    print(f"总结生成错误: {e}")
                finally:
                    result["completed"] = True

            # 启动API调用线程
            api_thread = threading.Thread(target=call_api)
            api_thread.daemon = True
            api_thread.start()

            # 等待结果，最多30秒
            timeout = 30  # 30秒超时
            start_time = time.time()
            while time.time() - start_time < timeout:
                if result.get("completed", False):
                    break
                time.sleep(0.5)  # 轻微暂停，减少CPU占用

            # 检查是否超时
            if not result.get("completed", False):
                result["timeout"] = True
                print(f"记忆 #{memory_id} 关键句子总结超时")

            # 处理结果
            key_sentence = result["key_sentence"]

            # 如果总结失败或为空，使用备选方法
            if not key_sentence or not result["success"] or result["timeout"]:
                print(f"记忆 #{memory_id} 使用备选方法提取关键句子")
                # 简单提取第一句话
                sentences = thought.split('。')
                if sentences:
                    key_sentence = sentences[0].strip()[:100]
                    if len(key_sentence) == 100:
                        key_sentence += "..."
                else:
                    key_sentence = thought[:100].strip()
                    if len(thought) > 100:
                        key_sentence += "..."

            # 更新记忆中的关键句子
            if 0 <= memory_id < len(self.memories):
                # 为关键句子生成新的嵌入向量
                print(f"为记忆 #{memory_id} 更新关键句子: {key_sentence}")

                try:
                    embedding_response = self.client.create_embedding(key_sentence, self.embedding_model)
                    if embedding_response and embedding_response["data"]:
                        new_embedding = embedding_response["data"][0]["embedding"]

                        # 从索引中移除旧的嵌入
                        if self.index is not None:
                            # FAISS不支持直接删除，所以我们需要重建索引
                            self._rebuild_index_without(memory_id)

                        # 更新记忆
                        self.memories[memory_id]["key_sentence"] = key_sentence
                        self.memories[memory_id]["embedding"] = new_embedding
                        self.memories[memory_id]["is_summarized"] = True

                        # 添加更新后的嵌入到索引
                        if self.index is not None:
                            self.index.add(np.array([new_embedding], dtype=np.float32))

                        # 保存更新后的记忆
                        self.save_memories()
                        print(f"记忆 #{memory_id} 关键句子总结已完成并更新")
                    else:
                        print(f"记忆 #{memory_id} 更新嵌入向量失败")
                except Exception as e:
                    print(f"更新记忆 #{memory_id} 嵌入向量时出错: {e}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"异步总结记忆失败: {e}\n{error_trace}")

    def _rebuild_index_without(self, exclude_id):
        """重建索引，排除指定ID的记忆"""
        if self.index is None or self.dimension is None:
            return

        # 创建新索引
        new_index = faiss.IndexFlatL2(self.dimension)

        # 添加除了exclude_id之外的所有嵌入
        embeddings = []
        for i, memory in enumerate(self.memories):
            if i != exclude_id:
                embeddings.append(np.array(memory["embedding"], dtype=np.float32))

        if embeddings:
            new_index.add(np.array(embeddings))

        # 更新索引
        self.index = new_index

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
