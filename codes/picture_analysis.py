import os
from datetime import datetime, timedelta
from collections import Counter

def analyze_image_files(directory: str):
    """
    分析指定目录下的图片文件名，并生成中断时长的排行榜。

    图片命名格式: YYYYMMDD_HHMMSS (例如: 20250615_140600)
    """
    print(f"--- 开始分析文件夹: {directory} ---")

    if not os.path.isdir(directory):
        print(f"错误: 文件夹 '{directory}' 不存在。")
        return

    # --- 第1步 & 第2步: 读取、去重并排序 ---
    print("步骤 1 & 2: 正在读取、去重并排序文件...")
    
    unique_files = set()
    duplicates = []
    all_file_info = []
    
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    filename = entry.name
                    base_name, _ = os.path.splitext(filename)

                    if base_name in unique_files:
                        duplicates.append(filename)
                    else:
                        unique_files.add(base_name)
                        try:
                            dt_object = datetime.strptime(base_name, '%Y%m%d_%H%M%S')
                            all_file_info.append({'datetime': dt_object, 'name': filename})
                        except ValueError:
                            # 忽略格式不正确的文件
                            pass
    except OSError as e:
        print(f"错误: 无法访问文件夹 '{directory}': {e}")
        return
        
    if not all_file_info:
        print("错误: 文件夹中没有找到符合命名格式的图片文件。")
        return

    all_file_info.sort(key=lambda x: x['datetime'])
    print(f"步骤 1 & 2 完成。共找到 {len(all_file_info)} 个有效文件，发现 {len(duplicates)} 个重名文件。")


    # --- 第3步: 分析时间序列，找出中断 ---
    print("\n步骤 3: 正在分析时间序列...")
    
    interruptions = []
    expected_delta = timedelta(minutes=6)
    expected_delta1 = timedelta(minutes=12) # 允许12分钟的间隔

    for i in range(1, len(all_file_info)):
        prev_file = all_file_info[i-1]
        current_file = all_file_info[i]
        time_difference = current_file['datetime'] - prev_file['datetime']

        if time_difference > timedelta(0) and time_difference not in [expected_delta, expected_delta1]:
            interruptions.append({
                'before': prev_file['name'],
                'after': current_file['name'],
                'duration': time_difference
            })
    print("步骤 3 完成。")


    # --- 第4步: 生成排行榜并输出报告 ---
    print("\n--- 分析结果报告 ---")

    print(f"\n[1] 重名文件总数: {len(duplicates)} 个")

    print(f"\n[2] 时间中断分析:")
    if not interruptions:
        print("  恭喜！未发现任何时间中断。")
    else:
        print(f"  总共发生 {len(interruptions)} 次中断 (即时间间隔不为6或12分钟)。")
        
        # --- 排行榜 1: 中断频率排行 ---
        print("\n  --- 中断频率排行榜 (按发生次数) ---")
        # 使用 Counter 统计每种时长出现的次数
        duration_counts = Counter(item['duration'] for item in interruptions)
        # 按照出现次数从多到少排序
        sorted_interruptions_by_count = duration_counts.most_common()
        
        # 显示排名前15的频率
        display_limit = 15
        for i, (duration, count) in enumerate(sorted_interruptions_by_count):
            if i >= display_limit:
                print(f"    ... (及其他 {len(sorted_interruptions_by_count) - display_limit} 种) ...")
                break
            print(f"    Top {i+1}: 时长 <{str(duration)}>  发生了 {count} 次")

        # --- 排行榜 2: 中断时长排行 ---
        print("\n  --- 最长中断时长排行榜 (按单次中断时间长度) ---")
        # 对中断按时长进行排序，找出最长的中断
        interruptions_sorted_by_length = sorted(interruptions, key=lambda x: x['duration'], reverse=True)

        # 显示排名前15的中断
        for i, item in enumerate(interruptions_sorted_by_length):
            if i >= display_limit:
                print(f"    ... (及其他 {len(interruptions_sorted_by_length) - display_limit} 次中断) ...")
                break
            print(f"    Top {i+1}: 持续 <{item['duration']}>")
            print(f"           (从 {item['before']} 到 {item['after']})")

    print("\n--- 分析结束 ---")


if __name__ == '__main__':
    # ===============================================================
    # 请确保这里的路径是正确的
    target_folder = 'D:\\2_Study\\数据分析\\weather_radar_huanan_final'
    # =================================_==============================
    
    analyze_image_files(target_folder)