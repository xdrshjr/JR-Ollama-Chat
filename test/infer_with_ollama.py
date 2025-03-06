import requests
import json
import time


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


def run_with_ollama():
    """测试Ollama API的函数"""
    client = OllamaClient()

    # 测试非流式响应
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你是谁?"}
    ]

    # print("非流式响应测试:")
    # response = client.chat_completion(
    #     model="qwq:latest",  # 替换为你的Ollama可用模型名称
    #     messages=messages,
    #     stream=False
    # )
    #
    # if response:
    #     print(f"响应ID: {response['id']}")
    #     print(f"模型: {response['model']}")
    #     print(f"内容: {response['choices'][0]['message']['content']}")
    # else:
    #     print("获取响应失败")

    # 测试流式响应
    print("\n流式响应测试:")
    stream_response = client.chat_completion(
        model="qwq:latest",  # 替换为你的Ollama可用模型名称
        messages=messages,
        stream=True
    )

    full_content = ""
    for chunk in stream_response:
        if chunk:
            content_delta = chunk["choices"][0]["delta"].get("content", "")
            full_content += content_delta
            print(content_delta, end="", flush=True)

    print("\n\n流式响应完成")


if __name__ == "__main__":
    run_with_ollama()