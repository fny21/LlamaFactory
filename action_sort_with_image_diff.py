import argparse
import os
import json
import random
from tqdm import tqdm
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)
from peft import PeftModel
from deepseek_api import DeepseekClient
from qwen_api import QwenClient
import json
from datetime import datetime

def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def generate_with_messages(model, processor, messages, images=None, max_new_tokens=2048):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if images is None:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )

    inputs = inputs.to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[-1] :]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text.strip()


def generate_text_with_messages(
    model,
    tokenizer,
    messages,
    max_new_tokens=2048,
    enable_thinking=True,
    do_sample=True,
    temperature=0.2,
):
    """
    仿照 generate_with_messages（VL）写的纯语言模型推理函数
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,   # 与 Qwen3-8B 官方示例一致
    )

    # breakpoint()

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
    )

    # breakpoint()

    # 放到模型所在设备（这里会是 cuda:1）
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text.strip()


# ===================== 主程序 =====================
def main(args):
    # 1. 加载 vLLM 模型和处理器
    model_name_or_path = args.model_name_or_path  # 你的 Qwen3-VL-8B-Instruct 基础模型路径
    llm_model_name_or_path = args.llm_model_name_or_path
    lora_path = args.lora_path                    # 你的 LoRA checkpoint 路径（checkpoint-1700）

    if args.use_qwen_vl_api:
        model = QwenClient()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
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

    if args.use_another_language_model:
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path, trust_remote_code=True)

        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:1"},   # 强制整个模型到 cuda:1
            trust_remote_code=True,
        )

        llm_model.eval()
    elif args.use_another_language_model_api:
        api_model = DeepseekClient()

    # 2. 初始化统计变量
    processed_num = 0
    correct_num = 0

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

    for input_file in tqdm(sampled_data[args.continue_from:], total=len(sampled_data[args.continue_from:])):

        images = input_file["images"]
        image_paths = [os.path.join(args.input_image_dir, image_path) for image_path in images]
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

        # 阶段1：逐张调用模型，仅比较“初始图 vs 当前图”。
        initial_image = raw_images[0]
        per_image_diffs = []
        for image_index in range(2, len(raw_images) + 1):
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

            if args.use_qwen_vl_api:
                input_content = [
                    {"type": "text", "text": compare_prompt},
                    {"type": "text", "text": "Image 1 (Initial):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{model.encode_image(image_paths[0])}"}},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{model.encode_image(image_paths[image_index-1])}"}},
                ]
                diff_output = model.chat_with_image(input_content=input_content)
                per_image_diffs.append((image_index-1, diff_output))
            else:
                compare_messages = [{"role": "user", "content": []}]
                compare_messages[0]["content"].append({"type": "text", "text": compare_prompt})
                compare_messages[0]["content"].append({"type": "text", "text": "Image 1 (Initial):"})
                compare_messages[0]["content"].append({"type": "image"})
                compare_messages[0]["content"].append({"type": "text", "text": f"Image 2:"})
                compare_messages[0]["content"].append({"type": "image"})

                diff_output = generate_with_messages(
                    model,
                    processor,
                    compare_messages,
                    images=[initial_image, raw_images[image_index - 1]],
                    max_new_tokens=args.compare_max_new_tokens,
                )
                if "</think>" in diff_output:
                    diff_output = diff_output.split("</think>")[-1]
                else:
                    diff_output = "Too complex, not known now."
                per_image_diffs.append((image_index-1, diff_output))
        
        # ================= 保存 image_diff 信息（与图片同级目录） =================
        if args.use_qwen_vl_api:
            image_diff_data = {
                "version": 1,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "sample_output_dir": sample_output_dir,
                "initial_image": {
                    "index": 1,
                    "saved_name": "image_01_initial.png",
                    "original_path": images[0] if len(images) > 0 else None,
                },
                "images": [
                    {
                        "index": idx,
                        "saved_name": ("image_01_initial.png" if idx == 1 else f"image_{idx:02d}.png"),
                        "original_path": img_path,
                    }
                    for idx, img_path in enumerate(images, start=1)
                ],
                "diffs": [
                    {
                        "target_image_index": target_idx,   # 对应 Image 2 的序号（从1开始）
                        "target_saved_name": f"image_{target_idx+1:02d}.png",
                        "diff_text": diff_text,
                    }
                    for target_idx, diff_text in per_image_diffs
                ],
                "image_name_lines": image_name_lines,  # 兼容你已有的人类可读映射
            }

            image_diff_json_path = os.path.join(sample_output_dir, "image_diffs.json")
            with open(image_diff_json_path, "w", encoding="utf-8") as f:
                json.dump(image_diff_data, f, ensure_ascii=False, indent=2)

            print(f"[INFO] image diffs saved to: {image_diff_json_path}")
            continue
        # ======================================================================

        # 阶段2：只把逐图差异作为文本输入，不再输入图片，进行纯文本推理回答原问题。
        question_text = input_file.get("question", "").replace("You may organize your reasoning process freely, but you must explicitly indicate your final answer with \"Final Answer:\".", "")
        observations_text = "\n\n".join(
            [f"Image {image_index} vs Initial Image:\n{diff_text}" for image_index, diff_text in per_image_diffs]
        )
        final_prompt = (
            "You are solving a robotic-arm visual reasoning question, but now I will only provide you with textual observations.\n"
            f"Question:\n{question_text}\n\n"
            "Note that I have already provided detailed annotations of the differences between each candidate image "
            "and the initial image. I will no longer provide the images themselves, but instead give you key parts "
            "of each image described in text form. Please determine the correct order of these images based on the key "
            "differences between each image and the original image, and use pure text reasoning to solve this.\n"
            "Note that the descriptions of the images may not be entirely accurate, and some small objects may be "
            "misidentified (for example, a pear may be mistaken for a lemon). You should focus on the color and "
            "shape information of the objects described in the text, and comprehensively analyze the descriptions "
            "of the four images to infer the most likely correct answer.\n"
            "Observations (each candidate image compared with the initial image):\n"
            f"{observations_text}\n\n"
            "You must explicitly indicate your final answer with \"Final Answer:\"."
        )

        final_messages = [{"role": "user", "content": []}]
        final_messages[0]["content"].append({"type": "text", "text": final_prompt})

        if args.use_another_language_model:
            model_output = generate_text_with_messages(
                llm_model,
                llm_tokenizer,
                [{"role": "user", "content": final_prompt}],
                max_new_tokens=args.final_max_new_tokens,
            )
        elif args.use_another_language_model_api:
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    model_output = api_model.chat(user_content=final_prompt)
                    break  # 成功则退出循环
                except Exception as e:
                    if attempt == max_retries - 1:  # 最后一次尝试仍失败
                        raise Exception(f"经过 {max_retries} 次尝试后仍然失败，最后错误: {e}") from e
                    else:
                        print(f"第 {attempt + 1} 次尝试失败: {e}，正在重试...")
                        continue
        else:
            model_output = generate_with_messages(
                model,
                processor,
                final_messages,
                images=None,
                max_new_tokens=args.final_max_new_tokens,
            )

        # 计算正确答案
        gt_answer = str(input_file['gt_answer'])
        model_answer = model_output.split("Final Answer:")[-1].strip().split("</think>")[-1].strip()
        correctness = gt_answer in model_answer

        with open(os.path.join(sample_output_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"question: {final_prompt}\n"
                f"final_model_output:\n{model_output}\n"
                f"gt_answer: {gt_answer}"
                f"correctness: {correctness}"
            )

        processed_num += 1
        if correctness:
            correct_num += 1
    
    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"total_samples_in_file: {len(data_list)}\n"
            f"start_from: {args.start_from}\n"
            f"sample_size_requested: {args.sample_size}\n"
            f"sample_size_actual: {len(sampled_data)}\n"
            f"sample_seed: {args.sample_seed}\n"
            f"processed: {processed_num}\n"
            f"correct: {correct_num}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--llm_model_name_or_path", default="/dataHW/workspace/fengningya/models/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径（如 checkpoint-1700）")
    parser.add_argument('--input_path', type=str, default='/dataHW/workspace/fengningya/vlm_task/new_len4mul1_rewrite.jsonl')
    parser.add_argument('--input_image_dir', type=str, default='/dataHW/workspace/fengningya/vlm_task')
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--compare_max_new_tokens', type=int, default=8192)
    parser.add_argument('--final_max_new_tokens', type=int, default=32768)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--continue_from', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='action_sort_with_image_diff')
    parser.add_argument('--other_info', type=str, default='image_diff_Qwen3-8B-instruct_zero_shot')
    parser.add_argument('--use_another_language_model', action="store_true")
    parser.add_argument('--use_another_language_model_api', action="store_true")
    parser.add_argument('--use_qwen_vl_api', action="store_true")
    args = parser.parse_args()
    if args.use_another_language_model and args.use_another_language_model_api:
        raise ValueError("`--use_another_language_model` and `--use_another_language_model_api` cannot be both set.")
    main(args)


'''
CUDA_VISIBLE_DEVICES=0 python action_sort_with_image_diff.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Thinking --use_another_language_model_api --other_info image_diff_Qwen3-VL-8B-think_final_answer_deepseek

CUDA_VISIBLE_DEVICES=1 python action_sort_with_image_diff.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct --use_another_language_model --other_info image_diff_Qwen3-VL-8B-instruct_final_answer_Qwen3-1-7B_zero_shot

CUDA_VISIBLE_DEVICES=2 python action_sort_with_image_diff.py --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Thinking --other_info image_diff_Qwen3-8B-Thinking_zero_shot_fix
'''