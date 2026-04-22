import os

# ====================== 请修改这里 ======================
# 你的txt文件所在的文件夹路径（Windows示例：D:\\files  或  Mac/Linux示例：/home/user/files）
FOLDER_PATH = "/dataHW/workspace/fengningya/LlamaFactory/action_sort_eval_results/Qwen3-8B-zero_shot"  # 默认为当前脚本所在文件夹
# ========================================================

correct_count = 0   # 正确数量
total_count = 0     # 总文件数量

# 遍历文件夹中的所有文件
for filename in os.listdir(FOLDER_PATH):
    # 只处理 .txt 文件
    if filename.endswith(".txt") and filename != "sum.txt":
        file_path = os.path.join(FOLDER_PATH, filename)
        
        try:
            # 读取文件所有行
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # 确保文件至少有2行（倒数第二行存在）
                if len(lines) >= 2:
                    # 获取倒数第二行，并去除空格/换行符
                    second_last_line = lines[-1].strip()
                    
                    # 判断是否为正确
                    if "correctness: True" in second_last_line:
                        correct_count += 1
                        total_count += 1
                    elif "correctness: False" in second_last_line:
                        total_count += 1
        except:
            # 跳过无法读取的文件
            pass

# 计算正确率
if total_count > 0:
    accuracy = (correct_count / total_count) * 100
else:
    accuracy = 0

# 输出结果
print("="*50)
print(f"统计完成！")
print(f"总文件数：{total_count}")
print(f"正确数：{correct_count}")
print(f"错误数：{total_count - correct_count}")
print(f"正确率：{accuracy:.2f}%")
print("="*50)