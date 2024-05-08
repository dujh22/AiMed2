#!/usr/bin/env python3
# coding: utf-8

# 本脚本适用于pytho3.10及以上
# requests和json是本脚本的依赖，需要安装
import requests
import json

url = 'http://192.168.0.118:8088/base'
header = {
    'Content-Type': 'application/json'
}
data = {
    'model' : "AiMed",
    'messages' : [
        {"role": "user", "content": "糖尿病随访方案"}
    ],
    'stream' : True,
    'temperature' : 1
}

response = requests.post(url, headers=header, json=data, stream=True)
for chunk in response.iter_content(chunk_size=1024): # 注意chunk是bytes类型，需要先转换
    # 处理响应内容
    # 将bytes类型转换为字符串类型
    str_obj = chunk.decode('utf-8')
    str_obj = str_obj.replace('data: ', '').strip()
    # 将字符串类型转换为json格式
    json_obj = json.loads(str_obj)

    # print(json_obj) # 本行是整个json脚本，下面是直接解析到对应回复的解析
    # 或者把上面一行注释掉，或者把下面三行注释调
    if 'content' in json_obj['choices'][0]['delta'].keys():
        print(json_obj['choices'][0]['delta']['content'], end="", flush=True)  # 同一行流式结果
        # print(json_obj['choices'][0]['delta']['content'], end="\n", flush=True) # 不同行流式结果
