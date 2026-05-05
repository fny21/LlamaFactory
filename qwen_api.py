# qwen_client.py
import os
from openai import OpenAI
import base64

class QwenClient:
    """通义千问API客户端封装类"""
    
    def __init__(self, model="qwen3-vl-plus"):
        """
        初始化客户端
        
        Args:
            api_key: API密钥，如果不提供则从环境变量DASHSCOPE_API_KEY读取
            model: 模型名称，默认qwen-vl-plus，可选其他模型
        """
        self.api_key = "sk-c5ae92c2451f4608a152102ce74dd059" or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请提供API_KEY或设置环境变量DASHSCOPE_API_KEY")
        
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def chat_with_image(self, input_content, model=None, max_retries=20):
        model = model or self.model
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": input_content}]
                )
                return self._extract_content(completion.model_dump_json())
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"错误: {e}"
                print(f"重试 {attempt+1}/{max_retries} 错误: {e}")

    def _extract_content(self, response_json):
        import json
        response_dict = json.loads(response_json) if isinstance(response_json, str) else response_json
        
        if "error" in response_dict:
            raise Exception(response_dict["error"])
        
        return response_dict["choices"][0]["message"]["content"]
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    

if __name__ == "__main__":
    model = QwenClient()
    image1_path = "/dataHW/workspace/fengningya/LlamaFactory/action_sort_with_image_diff/image_diff_Qwen3-VL-8B-think_final_answer_deepseek/task_327_ep651297_ordering_full/image_01_initial.png"
    image2_path = "/dataHW/workspace/fengningya/LlamaFactory/action_sort_with_image_diff/image_diff_Qwen3-VL-8B-think_final_answer_deepseek/task_327_ep651297_ordering_full/image_03.png"

    base64_image1 = model.encode_image(image1_path)
    base64_image2 = model.encode_image(image2_path)

    compare_prompt = (
                    "You will see two images, which are screenshots of a robotic arm's operation process. "
                    "The first image is the initial image, and the second image represents a scene after "
                    "the robotic arm has completed several operations. Please carefully compare the second image"
                    " with the initial image and describe the differences in detail. "
                    "(Note: You should focus on the differences that indicate what action the robotic arm has completed,"
                    " such as what is added to the bag or what the robotic arm is holding. Minor pixel differences due "
                    "to camera shake are not within the scope of consideration). "
                    "Note that many objects may look similar (such as pears, lemons, starfruits, etc.). "
                    "If you cannot accurately determine what a small object is, you should focus on describing its color, "
                    "shape, or other visual characteristics (for example, \"a yellow spherical object that looks like a lemon\")."
                    "Please output the key differences for the second image."
                )
    input_content = [
        {"type": "text", "text": compare_prompt},
        {"type": "text", "text": "Image 1 (Initial):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image1}"}},
        {"type": "text", "text": "Image 2:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image2}"}},
    ]
    print(model.chat_with_image(input_content=input_content))