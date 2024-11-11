from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os, json

app = Flask(__name__)

# 设置你的视频文件路径
VIDEO_FOLDER = 'XXXXXXXXX'

# 获取所有视频文件列表，按照阿拉伯数字顺序排序
def extract_numeric(filename):
    return int(os.path.splitext(filename)[0])

video_files = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')], key=extract_numeric)

# 视频加载路由，根据视频文件名从指定文件夹读取
@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

# 主页面路由，显示所有视频和评分表单
@app.route('/')
def index():
    return render_template('index.html', video_files=video_files)

# 用于存储评分记录
scores = []

# 路由处理打分提交
@app.route('/submit_score', methods=['POST'])
def submit_score():
    # 获取所有的评分
    for i, video_file in enumerate(video_files):
        score = request.form.get(f'score_{i+1}')
        scores.append({'video': video_file, 'score': score})

    # 保存评分结果到文件
    save_path = '/home/jianglifan/DPO-VDM/data/video/rain.json'
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=4)

    # 重定向到完成页面
    return render_template('finished.html')

if __name__ == '__main__':
    app.run(debug=True)
