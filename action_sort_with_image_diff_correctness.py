import os

# ====================== 请修改这里 ======================
# 你的txt文件所在的文件夹路径（Windows示例：D:\\files  或  Mac/Linux示例：/home/user/files）
FOLDER_PATH = "/dataHW/workspace/fengningya/LlamaFactory/action_sort_with_image_diff/image_diff_Qwen3-VL-8B-think_final_answer_deepseek"  # 默认为当前脚本所在文件夹
# FOLDER_PATH = "/dataHW/workspace/fengningya/LlamaFactory/action_sort_with_image_diff/image_diff_Qwen3-8B-Thinking_zero_shot_fix"
# ========================================================

# 总体统计
correct_count = 0
total_count = 0

# unknown_image_num == 0 统计
zero_unknown_correct = 0
zero_unknown_total = 0

# unknown_image_num != 0 统计
nonzero_unknown_correct = 0
nonzero_unknown_total = 0

# 遍历文件夹中的所有文件夹
for filename in os.listdir(FOLDER_PATH):
    if filename == "sum.txt":
        continue

    sample_dir = os.path.join(FOLDER_PATH, filename)
    if not os.path.isdir(sample_dir):
        continue

    for file in os.listdir(sample_dir):
        if file != "result.txt":
            continue

        file_path = os.path.join(sample_dir, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 统计 unknown_image_num
            unknown_image_num = sum(1 for line in lines if line == "Too complex, not known now.\n")

            if len(lines) < 1:
                print("empty file", file_path)
                continue

            # 你原代码这里其实取的是最后一行，不是倒数第二行；这里保持一致
            last_line = lines[-1].strip()

            is_correct = None
            if "correctness: True" in last_line:
                is_correct = True
            elif "correctness: False" in last_line:
                is_correct = False
            else:
                print("not found correctness", file_path)
                continue

            # 总体统计
            total_count += 1
            if is_correct:
                correct_count += 1

            # 分组统计
            if unknown_image_num == 0:
                zero_unknown_total += 1
                if is_correct:
                    zero_unknown_correct += 1
                print(f"unknown_image: {unknown_image_num}, {'correct' if is_correct else 'wrong'} (group: ==0)")
            else:
                nonzero_unknown_total += 1
                if is_correct:
                    nonzero_unknown_correct += 1
                print(f"unknown_image: {unknown_image_num}, {'correct' if is_correct else 'wrong'} (group: !=0)")

        except Exception as e:
            print(f"read error: {file_path}, err: {e}")
            continue

# 计算正确率
overall_acc = (correct_count / total_count * 100) if total_count > 0 else 0
zero_unknown_acc = (zero_unknown_correct / zero_unknown_total * 100) if zero_unknown_total > 0 else 0
nonzero_unknown_acc = (nonzero_unknown_correct / nonzero_unknown_total * 100) if nonzero_unknown_total > 0 else 0

# 输出结果
print("=" * 50)
print("统计完成！")
print(f"[总体] 总文件数：{total_count}, 正确数：{correct_count}, 错误数：{total_count - correct_count}, 正确率：{overall_acc:.2f}%")
print(f"[unknown_image_num == 0] 总数：{zero_unknown_total}, 正确数：{zero_unknown_correct}, 错误数：{zero_unknown_total - zero_unknown_correct}, 正确率：{zero_unknown_acc:.2f}%")
print(f"[unknown_image_num != 0] 总数：{nonzero_unknown_total}, 正确数：{nonzero_unknown_correct}, 错误数：{nonzero_unknown_total - nonzero_unknown_correct}, 正确率：{nonzero_unknown_acc:.2f}%")
print("=" * 50)