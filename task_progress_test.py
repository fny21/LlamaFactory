import argparse
import os
import json
import random
from tqdm import tqdm
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


Task_prompt = '''You are a capable agent designed to infer multi-step forward dynamics transitions in embodied decision-making. Your goal is to determine how many actions from a given ordered action sequence have already been completed by comparing an initial state image with a target current state image.

## Your Task
You will be provided with:

1. A single **Initial State Image**.
2. A single **Target Current State Image**.
3. An ordered list of exactly 4 actions.

The **Target Current State Image** is guaranteed to be the result of applying the first `k` actions from the ordered action sequence to the **Initial State Image**, where `k` is an integer from 1 to 4.

You must determine which action has most recently been completed.

Specifically:

1. Start from the **Initial State Image**.
2. Apply the actions in order, one at a time.
3. If the target image matches the state after applying Action `k`, then the answer is `k`.

## Important Notes

- The answer must be an integer from 1 to 4.
- The answer represents the number of actions that have already been completed.
- Do not output multiple possible answers.
- The target image corresponds to exactly one completed action step.

## Actions in Order

{actions_block}

## Output Format

Your response must end with exactly one line in the following format:

Final Answer: <integer>
'''


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


# ===================== 主程序 =====================
def main(args):
    # 1. 加载 vLLM 模型和处理器
    model_name_or_path = args.model_name_or_path  # 你的 Qwen3-VL-8B-Instruct 基础模型路径
    lora_path = args.lora_path                    # 你的 LoRA checkpoint 路径（checkpoint-1700）

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if lora_path is not None:
        print("Loading LoRA...")

        model = PeftModel.from_pretrained(
            model,
            lora_path
        )
        model = model.merge_and_unload()

    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    # 2. 初始化统计变量
    correct_num = 0
    wrong_num = 0

    # 3. 遍历输入文件
    data_list = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 去掉空行、空格
            if not line:
                continue
            # 把每行转成字典
            data = json.loads(line)
            data_list.append(data)
    
    output_info_path = os.path.join(args.output_path, args.other_info)
    os.makedirs(output_info_path, exist_ok=True)

    sample_size = min(args.sample_size, len(data_list))
    rng = random.Random(args.sample_seed)
    if sample_size < len(data_list):
        sampled_data = rng.sample(data_list, sample_size)
    else:
        sampled_data = data_list

    for input_file in tqdm(sampled_data, total=len(sampled_data)):
        original_image = Image.open(input_file["initial_image_path"]).convert("RGB")
        current_image = Image.open(input_file["current_image_path"]).convert("RGB")

        action_list = input_file["action_list"]
        actions_block = "\n".join(
            f"[Action {i}] {action}"
            for i, action in enumerate(action_list, start=1)
        )
        prompt = Task_prompt.format(actions_block=actions_block)
        
        # 构建 vLLM 输入格式：文本 + 图像路径/对象
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        messages[0]["content"].append(
            {"type": "text", "text": prompt + "\nCurrent State Image:"}
        )
        messages[0]["content"].append(
            {"type": "image"}
        )
        messages[0]["content"].append(
            {"type": "text", "text": "Future State Images:"}
        )
        messages[0]["content"].append(
            {"type": "image"}
        )

        # breakpoint()

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[original_image, current_image],
            return_tensors="pt",
            padding=True
        )

        inputs = inputs.to("cuda")

        # 模型推理
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=0.2
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        model_output = output_text.strip()

        # breakpoint()

        # 计算正确答案
        gt_answer = str(input_file['gt_answer'])
        model_answer = model_output.split("Final Answer:")[-1].strip().split("</think>")[-1].strip()
        correctness = gt_answer == model_answer

        with open(os.path.join(output_info_path, f"{input_file['current_image_path'].split('/')[-1].split('.')[0]}.txt"), "w", encoding="utf-8") as f:
            f.write(f"model_output: {model_output}\ngt_answer: {gt_answer}\ncorrectness: {correctness}\n")
        
        #  breakpoint()

        if correctness:
            correct_num += 1
        else:
            wrong_num += 1
    
    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"total_samples_in_file: {len(data_list)}\n"
            f"sample_size_requested: {args.sample_size}\n"
            f"sample_size_actual: {len(sampled_data)}\n"
            f"sample_seed: {args.sample_seed}\n"
            f"total: correct {correct_num}, wrong {wrong_num}, accuracy {correct_num/max(correct_num+wrong_num, 1)}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument('--input_path', type=str, default='/dataHW/workspace/fengningya/vlm_task/task_progress_len4.jsonl')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--output_path', type=str, default='task_progress_eval_results')
    parser.add_argument('--other_info', type=str, default='null')
    args = parser.parse_args()
    main(args)


'''
CUDA_VISIBLE_DEVICES=3 python task_progress_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Thinking --other_info Qwen3-VL-8B-Thinking-zero_shot
CUDA_VISIBLE_DEVICES=2 python task_progress_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct --other_info Qwen3-VL-8B-Instruct-zero_shot

'''