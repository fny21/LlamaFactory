import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_EXPERIMENTS = [
    "image_diff_local_qwen_final_deepseek",
    "image_diff_local_qwen_final_local_qwen",
]


@dataclass
class SampleRecord:
    sample_id: str
    model_answer_raw: str
    gt_answer_raw: str
    final_model_output: str
    correctness_raw: Optional[bool]
    candidate_answer: Optional[str]
    has_candidate_answer: bool
    suspect_truncated: bool
    is_correct: bool


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return ""
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def extract_line_value(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.*)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_correctness(raw: str) -> Optional[bool]:
    if raw == "True":
        return True
    if raw == "False":
        return False
    return None


def find_candidate_answer(model_answer_raw: str, final_model_output: str) -> Optional[str]:
    # 优先从已抽取的 model_answer 找 1~4 的候选步数
    m = re.search(r"\b([1-4])\b", model_answer_raw)
    if m:
        return m.group(1)

    # 再从完整输出里找 Final Answer: <integer>
    m = re.search(r"Final Answer:\s*([1-4])\b", final_model_output, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def detect_suspect_truncated(final_model_output: str, has_candidate_answer: bool, threshold_chars: int) -> bool:
    if has_candidate_answer:
        return False
    # 没有候选答案且文本很长，同时也没出现 Final Answer 标记，判为疑似长度截断
    if len(final_model_output) >= threshold_chars and "Final Answer:" not in final_model_output:
        return True
    return False


def parse_result_file(result_path: str, threshold_chars: int) -> SampleRecord:
    text = read_text(result_path)
    sample_id = os.path.basename(os.path.dirname(result_path))

    final_model_output = extract_between(text, "final_model_output:\n", "\n\nmodel_answer:")
    model_answer_raw = extract_line_value(text, "model_answer")
    gt_answer_raw = extract_line_value(text, "gt_answer")
    correctness_line = extract_line_value(text, "correctness")
    correctness_raw = parse_correctness(correctness_line)

    candidate_answer = find_candidate_answer(model_answer_raw, final_model_output)
    has_candidate_answer = candidate_answer is not None
    suspect_truncated = detect_suspect_truncated(final_model_output, has_candidate_answer, threshold_chars)

    # correctness 优先按文件记录；缺失时回退到 candidate_answer 与 gt_answer 比较
    if correctness_raw is not None:
        is_correct = correctness_raw
    else:
        is_correct = (candidate_answer is not None and candidate_answer == gt_answer_raw.strip())

    return SampleRecord(
        sample_id=sample_id,
        model_answer_raw=model_answer_raw,
        gt_answer_raw=gt_answer_raw,
        final_model_output=final_model_output,
        correctness_raw=correctness_raw,
        candidate_answer=candidate_answer,
        has_candidate_answer=has_candidate_answer,
        suspect_truncated=suspect_truncated,
        is_correct=is_correct,
    )


def collect_experiment_records(exp_dir: str, threshold_chars: int) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    if not os.path.isdir(exp_dir):
        return records

    for name in sorted(os.listdir(exp_dir)):
        if name == "sum.txt":
            continue
        sample_dir = os.path.join(exp_dir, name)
        if not os.path.isdir(sample_dir):
            continue
        result_path = os.path.join(sample_dir, "result.txt")
        if not os.path.isfile(result_path):
            continue
        try:
            records.append(parse_result_file(result_path, threshold_chars))
        except Exception as e:
            print(f"[WARN] parse failed: {result_path} | err: {e}")
    return records


def summarize_records(records: List[SampleRecord]) -> Dict[str, float]:
    total = len(records)
    correct = sum(1 for r in records if r.is_correct)
    no_candidate = sum(1 for r in records if not r.has_candidate_answer)
    with_candidate = total - no_candidate
    suspect_truncated = sum(1 for r in records if r.suspect_truncated)
    wrong = total - correct
    parse_by_fallback = sum(1 for r in records if r.correctness_raw is None)

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "with_candidate_answer": with_candidate,
        "without_candidate_answer": no_candidate,
        "suspect_truncated": suspect_truncated,
        "correctness_missing_and_fallback_used": parse_by_fallback,
    }


def print_summary(exp_name: str, summary: Dict[str, float]) -> None:
    total = int(summary["total"])
    correct = int(summary["correct"])
    wrong = int(summary["wrong"])
    with_candidate = int(summary["with_candidate_answer"])
    without_candidate = int(summary["without_candidate_answer"])
    suspect_truncated = int(summary["suspect_truncated"])
    fallback_used = int(summary["correctness_missing_and_fallback_used"])

    print("=" * 72)
    print(f"[Experiment] {exp_name}")
    print(f"Total: {total}")
    print(f"Correct: {correct} | Wrong: {wrong} | Accuracy: {summary['accuracy'] * 100:.2f}%")
    print(f"Has candidate answer: {with_candidate} | No candidate answer: {without_candidate}")
    print(f"Suspect truncated by length limit: {suspect_truncated}")
    print(f"Missing 'correctness' in file and fallback used: {fallback_used}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="task_progress_with_image_diff",
        help="Root output dir of task_progress_with_image_diff experiments.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment folder names under output_path.",
    )
    parser.add_argument(
        "--truncate_char_threshold",
        type=int,
        default=4000,
        help="When no candidate answer and output length >= threshold (and no 'Final Answer:'), mark as suspect truncated.",
    )
    args = parser.parse_args()

    all_records: List[SampleRecord] = []
    for exp in args.experiments:
        exp_dir = os.path.join(args.output_path, exp)
        records = collect_experiment_records(exp_dir, args.truncate_char_threshold)
        if not records:
            print(f"[WARN] No valid result.txt found in: {exp_dir}")
            continue
        all_records.extend(records)
        print_summary(exp, summarize_records(records))

    if all_records:
        print_summary("ALL_EXPERIMENTS", summarize_records(all_records))
    else:
        print("[ERROR] No records collected.")


if __name__ == "__main__":
    main()
