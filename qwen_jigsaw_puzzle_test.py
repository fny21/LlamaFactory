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


def generate_puzzle_prompt(n):
    input_string_1 = '''
You will solve an N×N image jigsaw reconstruction task.

The original image was cut into {K} = {N}×{N} tiles. I will provide the tiles in a fixed order, and each tile has a fixed ID.
All tiles keep their original orientation — no rotation and no flipping are allowed.

Tiles (IDs are fixed)

I will provide {K} tiles as:
'''.format(N=n, K=n*n)

    input_string_3 = '''
Task

Determine the correct placement of all tiles in an {N}×{N} grid so that they form a coherent and visually consistent original image.

Output rules (strict)

Output only a sequence of {K} integers representing tile IDs placed in row-major order (left to right, top to bottom):
Row 1: (Top-Left → Top-Right), ..., until Row {N}.

Use each tile ID from 1 to {K} exactly once.

Output no explanations, no reasoning, no additional text, and no punctuation other than spaces.

Output format example (format only)

'''.format(N=n, K=n*n)
    
    if n == 2:
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>
'''
    elif n == 3:
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>

Tile 5: <image>

Tile 6: <image>

Tile 7: <image>

Tile 8: <image>

Tile 9: <image>
'''
    else:
        assert n == 4
        input_string_2 = '''
Tile 1: <image>

Tile 2: <image>

Tile 3: <image>

Tile 4: <image>

Tile 5: <image>

Tile 6: <image>

Tile 7: <image>

Tile 8: <image>

Tile 9: <image>

Tile 10: <image>

Tile 11: <image>

Tile 12: <image>

Tile 13: <image>

Tile 14: <image>

Tile 15: <image>

Tile 16: <image>
'''

    if n == 2:
        input_string_4 = '''
a1 a2
a3 a4
'''
    elif n == 3:
        input_string_4 = '''
a1 a2 a3
a4 a5 a6
a7 a8 a9
'''
    else:
        assert n == 4
        input_string_4 = '''
a1 a2 a3 a4
a5 a6 a7 a8
a9 a10 a11 a12
a13 a14 a15 a16
'''
    return input_string_1 + input_string_2 + input_string_3 + input_string_4


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
    correct_num = [0, 0, 0]  # 2x2, 3x3, 4x4
    wrong_num = [0, 0, 0]

    # 3. 遍历输入文件
    input_files = sorted([f for f in os.listdir(args.input_path) if f.endswith(".json")])
    output_info_path = args.output_path
    os.makedirs(output_info_path, exist_ok=True)
    os.makedirs(os.path.join(output_info_path, "sample_record"), exist_ok=True)

    for index, input_file in tqdm(enumerate(input_files), total=len(input_files)):
        if index > args.max_samples:
            break

        # 读取数据
        with open(os.path.join(args.input_path, input_file), "r", encoding="utf-8") as f:
            data_item = json.load(f)

        original_img_path = data_item["original_image_path"]
        processing_steps = data_item["processing_steps"]
        puzzle_info = data_item["puzzle"]
        puzzle_size = puzzle_info["size"]
        shuffled_order = puzzle_info["shuffled_order"]

        # 处理图像
        img = Image.open(original_img_path)
        img = img.convert("RGB")
        for step in processing_steps:
            step_type = step["step_type"]
            if step_type == "crop":
                crop_bbox = step["crop_bbox"]
                img = img.crop(tuple(crop_bbox))
            elif step_type == "resize":
                resized_size = step["resized_size"]
                img = img.resize(resized_size, Image.Resampling.LANCZOS)

        img_w, img_h = img.size
        assert img_w == img_h
        if puzzle_size == 2:
            assert img_w == 140 * 2, f"{input_file} {img_w}"
        elif puzzle_size == 3:
            assert img_w == 140 * 3, f"{input_file} {img_w}"
        else:
            assert img_w == 140 * 4, f"{input_file} {img_w}"

        # 生成原始顺序的子图列表
        piece_size = 140
        original_pieces = []
        for idx in range(puzzle_size * puzzle_size):
            row = idx // puzzle_size
            col = idx % puzzle_size
            left = col * piece_size
            top = row * piece_size
            right = left + piece_size
            bottom = top + piece_size
            piece = img.crop((left, top, right, bottom))
            piece_rgb = pil_img2rgb(piece)
            original_pieces.append(piece_rgb)

        # 按 shuffled_order 调整顺序
        raw_images = [original_pieces[idx] for idx in shuffled_order]

        # 处理文本和图像交替输入
        puzzle_prompt = generate_puzzle_prompt(puzzle_size)
        text_list = puzzle_prompt.split('<image>')
        assert len(text_list) == len(raw_images) + 1

        # 构建 vLLM 输入格式：文本 + 图像路径/对象
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        for i in range(len(raw_images)):
            if text_list[i].strip():
                messages[0]["content"].append(
                    {"type": "text", "text": text_list[i]}
                )

            messages[0]["content"].append(
                {"type": "image"}
            )

        if text_list[-1].strip():
            messages[0]["content"].append(
                {"type": "text", "text": text_list[-1]}
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
                max_new_tokens=256,
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
        inverse_map = {original_idx: shuffled_pos for shuffled_pos, original_idx in enumerate(shuffled_order)}
        correct_order = list(range(puzzle_size * puzzle_size))
        answer = [inverse_map[idx] + 1 for idx in correct_order]
        matrix = [answer[i*puzzle_size : (i+1)*puzzle_size] for i in range(puzzle_size)]
        matrix_str = "\n".join([" ".join(map(str, row)) for row in matrix])

        # 判断正确性
        correctness = matrix_str.strip() == model_output.strip()

        # 保存结果
        with open(os.path.join(output_info_path, "sample_record", f"{input_file.split('.')[0]}_answer.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_output": model_output,
                "gt_answer": matrix_str,
                "correct": correctness
            }, f, ensure_ascii=False, indent=2)

        if correctness:
            correct_num[puzzle_size-2] += 1
        else:
            wrong_num[puzzle_size-2] += 1

        # 保存前30条样本可视化
        if index < 30:
            save_root = os.path.join(output_info_path, f"{input_file.split('.')[0]}_visualize")
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(f"{puzzle_prompt}\n\n{matrix_str}")
            for img_idx, one_img in enumerate(raw_images):
                img_name = f"{img_idx+1:02d}.png"
                one_img.save(os.path.join(save_root, img_name))

    # 输出统计结果
    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(f"2*2: correct {correct_num[0]}, wrong {wrong_num[0]}, accuracy {correct_num[0]/(correct_num[0]+wrong_num[0]) if correct_num[0]+wrong_num[0] > 0 else 0:.4f}\n")
        f.write(f"3*3: correct {correct_num[1]}, wrong {wrong_num[1]}, accuracy {correct_num[1]/(correct_num[1]+wrong_num[1]) if correct_num[1]+wrong_num[1] > 0 else 0:.4f}\n")
        f.write(f"4*4: correct {correct_num[2]}, wrong {wrong_num[2]}, accuracy {correct_num[2]/(correct_num[2]+wrong_num[2]) if correct_num[2]+wrong_num[2] > 0 else 0:.4f}\n")
        f.write(f"total: correct {sum(correct_num)}, wrong {sum(wrong_num)}, accuracy {sum(correct_num)/(sum(correct_num)+sum(wrong_num)) if sum(correct_num)+sum(wrong_num) > 0 else 0:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument("--input_path", type=str, required=True, help="输入 JSON 文件目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出结果目录")
    parser.add_argument("--max_samples", type=int, default=1000, help="最大测试样本数")
    args = parser.parse_args()
    main(args)


'''
CUDA_VISIBLE_DEVICES=2 python qwen_jigsaw_puzzle_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct --lora_path saves/Qwen3-VL-8B-Instruct/lora/train_2026-03-03-15-00-39/checkpoint-3000 --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/test --output_path eval_result/jigsaw_puzzle_Qwen3-VL-8B-Instruct_sft_step_3000 --max_samples 3000
CUDA_VISIBLE_DEVICES=3 python qwen_jigsaw_puzzle_test.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct --lora_path saves/Qwen3-VL-8B-Instruct/lora/train_2026-03-03-15-00-39/checkpoint-1500 --input_path /dataHW/workspace/fengningya/Bagel/puzzle_dataset/test --output_path eval_result/jigsaw_puzzle_Qwen3-VL-8B-Instruct_sft_step_1500 --max_samples 3000
'''