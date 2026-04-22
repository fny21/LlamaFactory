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

    for index, input_file in tqdm(enumerate(data_list), total=len(data_list)):
        if index > args.max_samples:
            break
        if index < args.start_from:
            continue

        images = input_file["images"]
        raw_images = [Image.open(os.path.join(args.input_image_dir, image_path)).convert("RGB") for image_path in images]

        # 处理文本
        puzzle_prompt = input_file["question"]
        
        # 构建 vLLM 输入格式：文本 + 图像路径/对象
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        messages[0]["content"].append(
            {"type": "text", "text": puzzle_prompt + "\nCurrent State Image:"}
        )
        messages[0]["content"].append(
            {"type": "image"}
        )
        messages[0]["content"].append(
            {"type": "text", "text": "Future State Images"}
        )

        for _ in range(len(raw_images)-1):
            messages[0]["content"].append(
                {"type": "image"}
            )

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=raw_images,
            return_tensors="pt",
            padding=True
        )

        inputs = inputs.to("cuda")


        # 模型推理
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.2
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        model_output = output_text.strip()

        # 计算正确答案
        gt_answer = str(input_file['gt_answer'])
        model_answer = model_output.split("Final Answer:")[-1].strip()
        correctness = gt_answer == model_answer

        with open(os.path.join(output_info_path, f"{input_file['id']}.txt"), "w", encoding="utf-8") as f:
            f.write(f"model_output: {model_output}\ngt_answer: {gt_answer}\ncorrectness: {correctness}\n")
        
        #  breakpoint()

        if correctness:
            correct_num += 1
        else:
            wrong_num += 1
    
    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(f"total: correct {correct_num}, wrong {wrong_num}, accuracy {correct_num/max(correct_num+wrong_num, 1)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument('--input_path', type=str, default='/dataHW/workspace/fengningya/vlm_task/new_len4mul1_rewrite.jsonl')
    parser.add_argument('--input_image_dir', type=str, default='/dataHW/workspace/fengningya/vlm_task')
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='action_sort_eval_results')
    parser.add_argument('--other_info', type=str, default='null')
    args = parser.parse_args()
    main(args)


'''
CUDA_VISIBLE_DEVICES=2 python action_sort_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct --other_info Qwen3-8B-zero_shot --start_from 4000
'''