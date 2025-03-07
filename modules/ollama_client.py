import json
import time
import requests
import numpy as np


class OllamaClient:
    def __init__(self, base_url="http://192.168.1.114:11434"):
        self.base_url = base_url

    def chat_completion(self, model="llama3", messages=None, stream=False, temperature=0.7, max_tokens=2048):
        """
        模拟OpenAI的chat.completions.create调用方式
        """
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        print(f"发送聊天请求到 {url}, 模型: {model}, 流式: {stream}")
        print(f"消息内容: {messages[0]['content'][:50]}..." if messages else "无消息")

        try:
            if stream:
                return self._stream_response(url, payload)
            else:
                return self._complete_response(url, payload)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"聊天请求失败: {e}\n{error_trace}")
            return None

    def _complete_response(self, url, payload):
        """处理非流式响应"""
        try:
            print(f"开始发送聊天请求: {url}")
            response = requests.post(url, json=payload, timeout=30)  # 添加超时
            response.raise_for_status()
            result = response.json()

            print(f"收到响应: 状态码 {response.status_code}")
            print(f"响应数据类型: {type(result)}")
            print(f"响应键: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

            # 检查响应格式并进行适当处理
            if not isinstance(result, dict):
                print(f"错误: 响应不是字典类型: {result}")
                return None

            # 适配不同的API格式
            # Ollama API格式
            if "message" in result and "content" in result["message"]:
                message_content = result["message"]["content"]
                # 转换为OpenAI格式
                formatted_response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": message_content
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": result.get("usage", {})
                }
                return formatted_response
            # 可能已经是OpenAI格式
            elif "choices" in result:
                return result
            else:
                print(f"警告: 未知的响应格式: {result}")
                # 尝试创建一个基本的兼容格式
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(result)  # 使用整个响应作为内容
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {}
                }
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            return None
        except ValueError as e:
            print(f"JSON解析错误: {e}")
            return None
        except Exception as e:
            import traceback
            print(f"未预期的错误: {e}\n{traceback.format_exc()}")
            return None

    def _stream_response(self, url, payload):
        """处理流式响应"""
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        content = data.get("message", {}).get("content", "")

                        # 格式化为类似OpenAI的流式响应格式
                        chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": payload["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None if not data.get("done", False) else "stop"
                                }
                            ]
                        }

                        yield chunk

                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        print(f"无法解析JSON: {line}")
                        continue

        except requests.exceptions.RequestException as e:
            print(f"流式请求错误: {e}")
            yield None

    def create_embedding(self, text, model="bge-m3"):
        """
        使用Ollama的嵌入API创建文本嵌入
        支持单个文本或文本列表
        """
        url = f"{self.base_url}/api/embeddings"

        if isinstance(text, list):
            # 批量处理多个文本
            embeddings = []
            for t in text:
                emb = self._get_single_embedding(t, model, url)
                if emb is not None:
                    embeddings.append(emb)
            return {
                "data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)],
                "model": model
            }
        else:
            # 处理单个文本
            embedding = self._get_single_embedding(text, model, url)
            if embedding is not None:
                return {
                    "data": [{"embedding": embedding, "index": 0}],
                    "model": model
                }
            return None

    def _get_single_embedding(self, text, model, url):
        """获取单个文本的嵌入向量"""
        payload = {
            "model": model,
            "prompt": text,
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # 返回嵌入向量
            embedding = result.get("embedding", [])

            # 确保维度一致性 - 如果已经初始化过索引，确保新的嵌入与索引维度一致
            if hasattr(self, 'dimension') and self.dimension:
                if len(embedding) > self.dimension:
                    embedding = embedding[:self.dimension]  # 截断过长的嵌入
                elif len(embedding) < self.dimension:
                    embedding.extend([0] * (self.dimension - len(embedding)))  # 填充过短的嵌入

            return embedding
        except Exception as e:
            print(f"获取嵌入向量失败: {e}")
            return None