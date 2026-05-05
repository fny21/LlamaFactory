import argparse
import json
import os
import random
from datetime import datetime

import torch
from PIL import Image
from peft import PeftModel
from qwen_api import QwenClient
from deepseek_api import DeepseekClient
from tqdm import tqdm
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


Compare_prompt = '''You will see two images, which are screenshots of a robotic arm's operation process.

The first image is the Initial State Image. The second image is the Target Current State Image after the robotic arm has completed one or more actions.

Please carefully compare the second image with the first image and describe the action-relevant differences in detail.

Important requirements:
- Focus on changes that help infer which actions have already been completed.
- Describe objects that were moved, picked up, placed into a container, removed from a visible location, or newly held by the robotic arm.
- Ignore minor camera-shake, lighting, and pixel-level differences.
- Many objects may look similar. If you cannot accurately identify a small object, describe its color, shape, size, and location instead of guessing too confidently.
- Do not decide the final step number yet. Only output the visual differences between the two images.
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


def load_local_vl_model(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": args.vl_device},
        trust_remote_code=True,
    )

    if args.lora_path is not None:
        print("Loading LoRA...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    return model, processor


def generate_with_messages(model, processor, messages, images=None, max_new_tokens=2048, device="cuda:0"):
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

    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text.strip()


def strip_thinking_text(text):
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def extract_final_answer(model_output):
    answer_text = strip_thinking_text(model_output)
    if "Final Answer:" in answer_text:
        answer_text = answer_text.split("Final Answer:")[-1]

    answer_text = answer_text.strip()
    if not answer_text:
        return answer_text

    return answer_text.splitlines()[0].strip()


def build_actions_block(action_list):
    return "\n".join(
        f"[Action {i}] {action}"
        for i, action in enumerate(action_list, start=1)
    )


def build_local_compare_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": Compare_prompt},
                {"type": "text", "text": "Image 1 (Initial State Image):"},
                {"type": "image"},
                {"type": "text", "text": "Image 2 (Target Current State Image):"},
                {"type": "image"},
            ],
        }
    ]


def build_api_compare_content(qwen_client, initial_image_path, current_image_path):
    return [
        {"type": "text", "text": Compare_prompt},
        {"type": "text", "text": "Image 1 (Initial State Image):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{qwen_client.encode_image(initial_image_path)}"},
        },
        {"type": "text", "text": "Image 2 (Target Current State Image):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{qwen_client.encode_image(current_image_path)}"},
        },
    ]


def build_final_prompt(action_list, diff_text):
    actions_block = build_actions_block(action_list)
    task_text = Task_prompt.format(actions_block=actions_block)

    return (
        "You are solving a robotic-arm task-progress recognition problem, but now I will only provide textual "
        "observations instead of images.\n\n"
        f"{task_text}\n\n"
        "I have already compared the Target Current State Image with the Initial State Image. The visual-difference "
        "description may be imperfect, so reason conservatively from the ordered actions and the described visible "
        "state changes.\n\n"
        "Visual differences between the Target Current State Image and the Initial State Image:\n"
        f"{diff_text}\n\n"
        "Using only the action list and the visual-difference description, infer how many actions have already been "
        "completed. Your response must end with exactly one line formatted as:\n"
        "Final Answer: <integer>"
    )


def get_sample_id(input_file):
    current_path = input_file.get("current_image_path", "sample")
    return os.path.splitext(os.path.basename(current_path))[0]


def main(args):
    if args.use_qwen_vl_api:
        qwen_client = QwenClient()
        local_model = None
        processor = None
    else:
        qwen_client = None
        local_model, processor = load_local_vl_model(args)

    if args.use_deepseek_api:
        deepseek_client = DeepseekClient()
    else:
        deepseek_client = None
        if local_model is None:
            local_model, processor = load_local_vl_model(args)

    data_list = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))

    output_info_path = os.path.join(args.output_path, args.other_info)
    os.makedirs(output_info_path, exist_ok=True)

    candidate_data = data_list[args.start_from:]
    sample_size = min(args.sample_size, len(candidate_data))
    rng = random.Random(args.sample_seed)
    if sample_size < len(candidate_data):
        sampled_data = rng.sample(candidate_data, sample_size)
    else:
        sampled_data = candidate_data

    processed_num = 0
    correct_num = 0
    wrong_num = 0

    for input_file in tqdm(sampled_data[args.continue_from:], total=len(sampled_data[args.continue_from:])):
        initial_image_path = input_file["initial_image_path"]
        current_image_path = input_file["current_image_path"]
        initial_image = pil_img2rgb(Image.open(initial_image_path))
        current_image = pil_img2rgb(Image.open(current_image_path))

        sample_id = get_sample_id(input_file)
        sample_output_dir = os.path.join(output_info_path, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)

        initial_save_path = os.path.join(sample_output_dir, "image_01_initial.png")
        current_save_path = os.path.join(sample_output_dir, "image_02_current.png")
        initial_image.save(initial_save_path)
        current_image.save(current_save_path)

        if args.use_qwen_vl_api:
            compare_content = build_api_compare_content(qwen_client, initial_image_path, current_image_path)
            diff_output = qwen_client.chat_with_image(input_content=compare_content)
        else:
            diff_output = generate_with_messages(
                local_model,
                processor,
                build_local_compare_messages(),
                images=[initial_image, current_image],
                max_new_tokens=args.compare_max_new_tokens,
                device=args.vl_device,
            )

        diff_text = strip_thinking_text(diff_output)

        image_diff_data = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "sample_output_dir": sample_output_dir,
            "initial_image": {
                "saved_name": "image_01_initial.png",
                "original_path": initial_image_path,
            },
            "current_image": {
                "saved_name": "image_02_current.png",
                "original_path": current_image_path,
            },
            "diff_model": "qwen_api" if args.use_qwen_vl_api else "local_qwen_vl",
            "diff_raw_output": diff_output,
            "diff_text": diff_text,
        }
        with open(os.path.join(sample_output_dir, "image_diffs.json"), "w", encoding="utf-8") as f:
            json.dump(image_diff_data, f, ensure_ascii=False, indent=2)

        final_prompt = build_final_prompt(input_file["action_list"], diff_text)

        if args.use_deepseek_api:
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    model_output = deepseek_client.chat(user_content=final_prompt)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Deepseek API failed after {max_retries} retries. Last error: {e}") from e
                    print(f"Deepseek API attempt {attempt + 1} failed: {e}. Retrying...")
        else:
            final_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": final_prompt}],
                }
            ]
            model_output = generate_with_messages(
                local_model,
                processor,
                final_messages,
                images=None,
                max_new_tokens=args.final_max_new_tokens,
                device=args.vl_device,
            )

        gt_answer = str(input_file["gt_answer"])
        model_answer = extract_final_answer(model_output)
        correctness = gt_answer == model_answer

        with open(os.path.join(sample_output_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"final_prompt:\n{final_prompt}\n\n"
                f"final_model_output:\n{model_output}\n\n"
                f"model_answer: {model_answer}\n"
                f"gt_answer: {gt_answer}\n"
                f"correctness: {correctness}\n"
            )

        processed_num += 1
        if correctness:
            correct_num += 1
        else:
            wrong_num += 1

    with open(os.path.join(output_info_path, "sum.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"total_samples_in_file: {len(data_list)}\n"
            f"start_from: {args.start_from}\n"
            f"continue_from: {args.continue_from}\n"
            f"sample_size_requested: {args.sample_size}\n"
            f"sample_size_actual: {len(sampled_data)}\n"
            f"sample_seed: {args.sample_seed}\n"
            f"processed: {processed_num}\n"
            f"correct: {correct_num}\n"
            f"wrong: {wrong_num}\n"
            f"accuracy: {correct_num / max(processed_num, 1)}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen3-VL-8B-Instruct 基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径")
    parser.add_argument("--input_path", type=str, default="/dataHW/workspace/fengningya/vlm_task/task_progress_len4.jsonl")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--compare_max_new_tokens", type=int, default=8192)
    parser.add_argument("--final_max_new_tokens", type=int, default=32768)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--continue_from", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="task_progress_with_image_diff")
    parser.add_argument("--other_info", type=str, default="image_diff_Qwen3-VL-8B-Instruct_final_Qwen3-VL-8B-Instruct")
    parser.add_argument("--vl_device", type=str, default="cuda:0")
    parser.add_argument("--use_qwen_vl_api", action="store_true", help="Use Qwen API for stage-1 image-diff generation")
    parser.add_argument(
        "--use_deepseek_api",
        "--use_another_language_model_api",
        dest="use_deepseek_api",
        action="store_true",
        help="Use Deepseek API for stage-2 text-only reasoning",
    )
    args = parser.parse_args()
    main(args)


'''
Examples:

CUDA_VISIBLE_DEVICES=0 python task_progress_with_image_diff.py \
  --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct \
  --other_info image_diff_local_qwen_final_local_qwen

python task_progress_with_image_diff.py \
  --model_name_or_path /dataHW/workspace/fengningya/models/Qwen3-VL-8B-Instruct \
  --use_qwen_vl_api \
  --use_deepseek_api \
  --other_info image_diff_qwen_api_final_deepseek
'''
