# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

class DeepseekClient:
    def __init__(self, base_url: str = "https://api.deepseek.com"):
        """
        初始化 Deepseek 客户端
        :param api_key: 你的 Deepseek API Key
        :param base_url: API 地址（默认无需修改）
        """
        self.client = OpenAI(
            api_key="sk-082ff407e6f54716bf88a7a8922fb5b0",
            base_url=base_url
        )

    def chat(
        self,
        user_content: str,
        system_content: str = "You are a helpful assistant",
        model: str = "deepseek-v4-pro",
        stream: bool = False,
        reasoning_effort: str = "high",
        max_retries: int = 20
    ):
        """
        调用 Deepseek 对话接口
        :param user_content: 用户输入的问题
        :param system_content: 系统提示词
        :param model: 模型名称
        :param stream: 是否流式输出
        :param reasoning_effort: 推理力度
        :param max_retries: 最大重试次数
        :return: 模型返回的回答内容
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    stream=stream,
                    reasoning_effort=reasoning_effort,
                    extra_body={"thinking": {"type": "enabled"}}
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"错误: {e}"
                print(f"重试 {attempt+1}/{max_retries} 错误: {e}")
    

# client = DeepseekClient()
# answer = client.chat(user_content="Hello")
# print(answer)
