import argparse
import os
import json
import random
from tqdm import tqdm
from PIL import Image
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if args.lora_path != "none":
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

    input_files = os.listdir(args.input_path)
    input_files.sort()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "tensors"), exist_ok=True)

    for index, input_file in tqdm(enumerate(input_files)):
        with open(os.path.join(args.input_path, input_file), "r", encoding="utf-8") as f:
            data_item = json.load(f)
        original_img_path = data_item["original_image_path"]
        img = Image.open(original_img_path)
        img = img.convert("RGB")  # 初始转为RGB，和预处理一致
        
        # 获取图片原始宽高
        width, height = img.size
        print(f"原始图片分辨率：{width} x {height}")
        
        # 1. 判断长宽是否均大于140
        if width <= 140 or height <= 140:
            continue
        
        # 2. 计算中心裁剪的坐标（尽可能保留正中心）
        # 计算需要裁剪的起始坐标
        left = (width - 140) / 2
        top = (height - 140) / 2
        left = int(round(left))
        top = int(round(top))
        right = left + 140
        bottom = top + 140
        
        # 3. 执行中心裁剪
        cropped_img = img.crop((left, top, right, bottom))
        
        # 4. 保存裁剪后的图片
        original_image_name = original_img_path.split('/')[-1].split('.')[0]
        save_image_name = original_image_name + "_croped.png"
        cropped_img.save(os.path.join(args.output_path, "image", save_image_name))
        save_tensor_name = original_image_name + "_embed.pth"

        # 请续写
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": cropped_img}]
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # breakpoint()

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # breakpoint()

        pixel_values = inputs.pixel_values.to("cuda")

        with torch.no_grad():  # 禁用梯度计算，节省内存
            image_embeds = model.get_image_features(pixel_values, inputs.image_grid_thw, return_dict=True).pooler_output

            # breakpoint()

            image_embeds = image_embeds[0].to("cpu", torch.torch.bfloat16)

            # breakpoint()
            
            torch.save(image_embeds, os.path.join(args.output_path, "tensors", save_tensor_name))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, default="none", help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument("--input_path", type=str, required=True, help="输入 JSON 文件目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出结果目录")
    args = parser.parse_args()
    main(args)



'''
CUDA_VISIBLE_DEVICES=0 python vision_feature_extract.py --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/test --output_path /dataHW/workspace/fengningya/LlamaFactory/image_and_feature/pretrain/test --model_name_or_path /dataHW/workspace/fengningya/models/Qwen2-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=1 python vision_feature_extract.py --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/train --output_path /dataHW/workspace/fengningya/LlamaFactory/image_and_feature/pretrain/train --model_name_or_path /dataHW/workspace/fengningya/models/Qwen2-VL-7B-Instruct

CUDA_VISIBLE_DEVICES=2 python vision_feature_extract.py --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/test --output_path /dataHW/workspace/fengningya/LlamaFactory/image_and_feature/sft/test --model_name_or_path /dataHW/workspace/fengningya/models/Qwen2-VL-7B-Instruct --lora_path /dataHW/workspace/fengningya/LlamaFactory/saves/Qwen2-VL-7B-Instruct/lora/Qwen2-VL-7B-not-freeze-train_2026-03-28-12-30-53/checkpoint-3500
CUDA_VISIBLE_DEVICES=3 python vision_feature_extract.py --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/train --output_path /dataHW/workspace/fengningya/LlamaFactory/image_and_feature/sft/train --model_name_or_path /dataHW/workspace/fengningya/models/Qwen2-VL-7B-Instruct --lora_path /dataHW/workspace/fengningya/LlamaFactory/saves/Qwen2-VL-7B-Instruct/lora/Qwen2-VL-7B-not-freeze-train_2026-03-28-12-30-53/checkpoint-3500
'''
