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

        if stream:
            return self._stream_response(url, payload)
        else:
            return self._complete_response(url, payload)

    def _complete_response(self, url, payload):
        """处理非流式响应"""
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # 格式化为类似OpenAI的响应格式
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
                            "content": result.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": result.get("usage", {})
            }

            return formatted_response

        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
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
            return embedding
        except Exception as e:
            print(f"获取嵌入向量失败: {e}")
            return None