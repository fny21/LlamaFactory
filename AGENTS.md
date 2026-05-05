# Agent Notes: Key Evaluation Scripts

This repository contains several ad-hoc evaluation scripts at the repo root. The notes below document only the three key scripts that are currently important for this workspace:

- `action_sort_test.py`
- `action_sort_with_image_diff.py`
- `task_progress_test.py`

Do not assume these scripts are integrated with the normal LlamaFactory training/evaluation pipeline. They are standalone Python entrypoints for Qwen3-VL style visual reasoning experiments.

## Shared Patterns

All three scripts load JSONL input, sample a subset with `sample_size` and `sample_seed`, run model generation, parse `Final Answer:`, and write plain-text outputs under `<output_path>/<other_info>/`.

Common model behavior:

- Vision model class: `Qwen3VLForConditionalGeneration`.
- Processor: `AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)`.
- Optional LoRA: if `--lora_path` is provided, `PeftModel.from_pretrained(...)` is merged with `merge_and_unload()`.
- Generation is sampled with `do_sample=True` and `temperature=0.2`.
- The generated text after the prompt is decoded with `skip_special_tokens=True`.
- Thinking-model output is handled by splitting after `</think>` when extracting the final answer.

Common output pattern:

- Per-sample files include the raw model output, ground-truth answer, and a boolean correctness marker.
- `sum.txt` records sample count, seed, and aggregate counts.

## `action_sort_test.py`

Purpose: baseline direct visual reasoning for the action-sort task.

Expected input JSONL fields:

- `id`: sample id used for output filename.
- `images`: list of image paths relative to `--input_image_dir`.
- `question`: full prompt/question text for the action-sort puzzle.
- `gt_answer`: ground-truth answer.

Task semantics:

- `images[0]` is the initial/current state image.
- The remaining images are future state images whose order has been shuffled.
- The model receives the original question plus all images in one multimodal prompt.
- The script expects the model to end with `Final Answer: ...`.

Important implementation details:

- Images are loaded from `os.path.join(input_image_dir, image_path)`.
- Before inference, every image is resized to double width and double height.
- Prompt layout is:
  - question text plus `Current State Image:`
  - first image
  - `Future State Images:`
  - remaining images
- `max_new_tokens` is `32768`.
- Correctness is checked with substring containment: `str(gt_answer) in model_answer`, not exact equality.
- Per-sample result path: `<output_path>/<other_info>/<id>.txt`.
- Default output root: `action_sort_eval_results`.

Use this script when you want the VLM to solve the action-sort task directly from images in one pass.

## `action_sort_with_image_diff.py`

Purpose: two-stage action-sort pipeline that separates visual comparison from final ordering reasoning.

Expected input JSONL fields:

- `id`: sample id used as a subdirectory name.
- `images`: list of image paths relative to `--input_image_dir`.
- `question`: original action-sort question.
- `gt_answer`: ground-truth answer.

Task semantics:

- `images[0]` is the initial image.
- Each subsequent image is compared independently against the initial image.
- The final answer is inferred from textual differences rather than from the images directly.

Pipeline:

1. Stage 1, image-diff generation:
   - For each candidate image after the initial image, compare `initial image` vs `candidate image`.
   - Ask the VL model to describe action-relevant differences, such as objects added to the bag or held by the robotic arm.
   - The prompt explicitly says to ignore minor camera-shake pixel differences.
   - The prompt also warns that similar objects may be confused and asks for color/shape descriptions when object identity is uncertain.

2. Stage 2, text-only final reasoning:
   - Build `observations_text` from all per-image diff descriptions.
   - Remove the standard `Final Answer:` instruction sentence from the original question if present.
   - Ask a text model, API model, or the same VL model without images to infer the final action-sort answer.

Model modes:

- Default: local Qwen3-VL handles both image diffs and final text-only reasoning.
- `--use_another_language_model`: use a separate local causal LM for stage 2; defaults to `/dataHW/workspace/fengningya/models/Qwen3-1.7B` and places it on `cuda:1`.
- `--use_another_language_model_api`: use `DeepseekClient()` for stage 2, with up to 10 retries.
- `--use_qwen_vl_api`: use `QwenClient()` for stage 1 image diffs only; saves `image_diffs.json` and then `continue`s, so it does not run final reasoning or correctness.

Important implementation details:

- The local VL model is forced to `cuda:0` with `device_map={"": "cuda:0"}`.
- The optional local text model is forced to `cuda:1`.
- `--use_another_language_model` and `--use_another_language_model_api` are mutually exclusive.
- `--continue_from` resumes within the sampled data list after sampling has already happened.
- Each sample gets a directory: `<output_path>/<other_info>/<id>/`.
- The script saves copied input images in model-input order:
  - `image_01_initial.png`
  - `image_02.png`, `image_03.png`, etc.
- Default output root: `action_sort_with_image_diff`.
- Stage 1 token limit is `--compare_max_new_tokens`, default `8192`.
- Stage 2 token limit is `--final_max_new_tokens`, default `32768`.
- Correctness is checked with substring containment: `str(gt_answer) in model_answer`.

Important caveat:

- In local VL stage 1, if the diff output does not contain `</think>`, the diff text is replaced with `Too complex, not known now.`. This behavior is especially important when using non-thinking models, because valid non-thinking responses may be discarded.

Use this script when you want to inspect whether converting images into explicit differences helps final action-sort reasoning, or when you want to swap in a stronger text-only model/API for the final decision.

## `task_progress_test.py`

Purpose: evaluate task-progress recognition from an initial image, a target current image, and an ordered list of four actions.

Expected input JSONL fields:

- `initial_image_path`: path to the initial state image.
- `current_image_path`: path to the target current state image.
- `action_list`: ordered list of exactly 4 actions.
- `gt_answer`: ground-truth integer step as a string or number.

Task semantics:

- The target current image is guaranteed to be the result of applying the first `k` actions from `action_list` to the initial image.
- `k` is an integer from 1 to 4.
- The model must answer which action has most recently been completed.
- The final answer represents the number of completed actions.

Prompt construction:

- `Task_prompt` explains the task and inserts `action_list` as:
  - `[Action 1] ...`
  - `[Action 2] ...`
  - `[Action 3] ...`
  - `[Action 4] ...`
- The model receives two images in this order:
  - `original_image`
  - `current_image`
- The expected output must end with `Final Answer: <integer>`.

Important implementation details:

- Images are opened directly from `initial_image_path` and `current_image_path`.
- `sample_size` is sampled from the entire input file; there is no `start_from` argument in this script.
- `max_new_tokens` is `32768`.
- Correctness is strict equality: `str(gt_answer) == model_answer`.
- Per-sample result filename is derived from the basename of `current_image_path`.
- Default input path: `/dataHW/workspace/fengningya/vlm_task/task_progress_len4.jsonl`.
- Default output root: `task_progress_eval_results`.

Use this script for progress-index evaluation, not for shuffled future-image ordering.
