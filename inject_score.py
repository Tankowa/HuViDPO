import json
import os

# 读取评分文件
def load_scores(score_file_path):
    with open(score_file_path, 'r') as f:
        scores = json.load(f)
    return scores

# 将 video 和 score 注入到 json 文件中
def inject_scores(json_file_path, scores, json_index, NUM_PER_PROMPT):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 创建新的结构以包含 video 和 score
    for idx, entry in enumerate(data):
        # 对每个条目，注入对应的 video 和 score
        video_name = scores[json_index * NUM_PER_PROMPT + idx]["video"]
        score_value = scores[json_index * NUM_PER_PROMPT + idx]["score"]

        # 注入新的字段
        entry["video"] = video_name
        entry["score"] = score_value

    return data

# 合并所有的 json 文件
def merge_json_files(json_files):
    merged_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)  # 直接合并整个数据
    return merged_data

# 主函数
def main():
    # 文件路径和参数
    score_file_path = 'data/video/scores.json'  # 评分文件路径
    video_json_folder = 'data/video_json'  # json 文件的目录
    num_videos = 3  # 具体数量可以根据需求调整
    NUM_PER_PROMPT = 2  # 每个 json 文件中包含的视频数量
    output_file_path = 'data/merged_output.json'  # 合并后的输出路径

    # 读取评分文件
    scores = load_scores(score_file_path)

    # 遍历 json 文件并注入评分
    all_json_files = []
    for i in range(num_videos):
        json_file_path = os.path.join(video_json_folder, f'{i}.json')
        if os.path.exists(json_file_path):
            injected_data = inject_scores(json_file_path, scores, i, NUM_PER_PROMPT)
            with open(json_file_path, 'w') as f:
                json.dump(injected_data, f, indent=4)
            all_json_files.append(json_file_path)

    # 合并所有 json 文件
    merged_data = merge_json_files(all_json_files)

    # 保存合并后的 json 文件
    with open(output_file_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f'所有 JSON 文件已成功合并并保存到 {output_file_path}')

if __name__ == '__main__':
    main()
