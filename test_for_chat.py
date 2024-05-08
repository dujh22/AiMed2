# 本脚本适用于pytho3.10及以上
# openai是本脚本的依赖，需要安装
# pip install openai
# 如果是不正确的openai脚本，可能需要再cchardet
# pip install cchardet

import openai
if __name__ == "__main__":
    # openai.api_base = "http://localhost:8000/v1"
    openai.api_base = "http://192.168.0.118:8088/base"
    openai.api_key = "none"
    for chunk in openai.ChatCompletion.create( # 参数详细见接口文档
        model="AiMed",
        messages=[
            {"role": "user", "content": "你是谁开发的"}
        ],
        stream=False,
        temperature=1
    ):
        # print(chunk) # 本行是整个json脚本，下面三行是直接解析到对应回复的解析
        # 或者把上面一行注释掉，或者把下面三行注释调
        if hasattr(chunk.choices[0].delta, "content"):
            print(chunk.choices[0].delta.content, end="", flush=True) # 同一行流式结果
            # print(chunk.choices[0].delta.content, end="\n", flush=True) # 不同行流式结果