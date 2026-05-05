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
    processed_num = 0

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

    candidate_data = data_list[args.start_from :]
    sample_size = min(args.sample_size, len(candidate_data))
    rng = random.Random(args.sample_seed)
    if sample_size < len(candidate_data):
        sampled_data = rng.sample(candidate_data, sample_size)
    else:
        sampled_data = candidate_data

    for input_file in tqdm(sampled_data, total=len(sampled_data)):

        images = input_file["images"]
        raw_images = [pil_img2rgb(Image.open(os.path.join(args.input_image_dir, image_path))) for image_path in images]

        sample_id = str(input_file["id"])
        sample_output_dir = os.path.join(output_info_path, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        # 按输入模型时的顺序保存图片，便于人工核对。
        image_name_lines = []
        for image_index, (image_path, image) in enumerate(zip(images, raw_images), start=1):
            if image_index == 1:
                save_name = f"image_{image_index:02d}_initial.png"
            else:
                save_name = f"image_{image_index:02d}.png"

            save_path = os.path.join(sample_output_dir, save_name)
            image.save(save_path)
            image_name_lines.append(f"{save_name} <- {image_path}")

        # 处理文本
        compare_prompt = "You will see a set of images, which are screenshots of a robotic arm's operation process. " \
        "The first image is the initial image, and each subsequent image represents a scene after the robotic arm " \
        "has completed several operations (except for the first image, the order of the images has been shuffled, " \
        "so the images do not necessarily appear in sequential order). For each image except the initial one, " \
        "please carefully compare it with the initial image and describe the differences in detail. " \
        "(Note: You should focus on the differences that indicate what action the robotic arm has completed, " \
        "such as what is added to the bag or what the robotic arm is holding. Minor pixel differences due to camera " \
        "shake are not within the scope of consideration). Please output the key differences for each image in sequence."
        
        # 构建 vLLM 输入格式：文本 + 图像路径/对象
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        messages[0]["content"].append({"type": "text", "text": compare_prompt})
        messages[0]["content"].append({"type": "text", "text": "Image 1 (Initial):"})
        messages[0]["content"].append({"type": "image"})

        for image_index in range(2, len(raw_images) + 1):
            messages[0]["content"].append({"type": "text", "text": f"Image {image_index}:"})
            messages[0]["content"].append({"type": "image"})

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

        with open(os.path.join(sample_output_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"sample_id: {input_file['id']}\n"
                f"question: {input_file.get('question', '')}\n"
                f"image_num: {len(images)}\n"
                f"saved_images:\n" + "\n".join(image_name_lines) + "\n"
                f"model_output:\n{model_output}\n"
            )

        processed_num += 1
    
    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"total_samples_in_file: {len(data_list)}\n"
            f"start_from: {args.start_from}\n"
            f"sample_size_requested: {args.sample_size}\n"
            f"sample_size_actual: {len(sampled_data)}\n"
            f"sample_seed: {args.sample_seed}\n"
            f"processed: {processed_num}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument('--input_path', type=str, default='/dataHW/workspace/fengningya/vlm_task/new_len4mul1_rewrite.jsonl')
    parser.add_argument('--input_image_dir', type=str, default='/dataHW/workspace/fengningya/vlm_task')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='action_sort_observe_output')
    parser.add_argument('--other_info', type=str, default='image_diff_Qwen3-8B-instruct_zero_shot')
    args = parser.parse_args()
    main(args)


'''
CUDA_VISIBLE_DEVICES=2 python action_sort_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct
'''