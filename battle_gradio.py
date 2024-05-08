import gradio as gr
import requests
import json

import uvicorn
from fastapi import FastAPI

# 定义全局变量
model_urls = {
    'AiMed': 'http://192.168.0.155:8000/v1/chat/completions',
    'Doctor': 'http://192.168.0.118:8088/base',
    'GPT4': 'http://192.168.0.118:2223/v1/chat/completions'
}

header = {
    'Content-Type': 'application/json'
}
data = {
    'model': "AiMed",
    'stream': True,
    'temperature': 1
}

web_path = "/arena2"

app = FastAPI()


# 定义模型请求函数，处理流式响应
def send_request(model, prompt):
    url = model_urls[model]
    data['messages'] = [{"role": "user", "content": prompt}]
    response = requests.post(url, headers=header, json=data, stream=True)
    answer = ""
    last_content = ""
    num = 0
    for chunk in response.iter_content(chunk_size=1024):
        str_obj = chunk.decode('utf-8').replace('data: ', '').strip()
        try:
            json_obj = json.loads(str_obj)
            if 'content' in json_obj['choices'][0]['delta'].keys():
                answer += json_obj['choices'][0]['delta']['content']
                if json_obj['choices'][0]['delta']['content'] == last_content:
                    num += 1
                    if num == 5:
                        break
                else:
                    last_content = json_obj['choices'][0]['delta']['content']
        except json.JSONDecodeError:
            continue
    return answer


# 创建 Gradio 界面
def gradio_interface(prompt, model1, model2=None, model3=None):
    response1 = send_request(model1, prompt) if model1 else ""
    response2 = send_request(model2, prompt) if model2 else ""
    response3 = send_request(model3, prompt) if model3 else ""
    return response1, response2, response3



# 使用 Blocks API 创建布局
with gr.Blocks(title="Med-Eval 模型时间竞技") as demo:
    gr.Markdown("# Med-Eval 模型竞技")

    with gr.Row():
        with gr.Column(scale=1):
            model1_dropdown = gr.Dropdown(list(model_urls.keys()), label="模型 1")
            output1 = gr.Textbox(label="模型 1 响应", interactive=False, lines=10)
        with gr.Column(scale=1):
            model2_dropdown = gr.Dropdown(list(model_urls.keys()), label="模型 2")
            output2 = gr.Textbox(label="模型 2 响应", interactive=False, lines=10)
        with gr.Column(scale=1):
            model3_dropdown = gr.Dropdown(list(model_urls.keys()), label="模型 3")
            output3 = gr.Textbox(label="模型 3 响应", interactive=False, lines=10)

    with gr.Row():
        input_textbox = gr.Textbox(label="输入", placeholder="请输入您的问题...")

    with gr.Row():
        submit_button = gr.Button("提交")
        clear_button = gr.Button("清空")

    # 当按钮被点击时，执行 gradio_interface 函数
    submit_button.click(
        gradio_interface,
        inputs=[input_textbox, model1_dropdown, model2_dropdown, model3_dropdown],
        outputs=[output1, output2, output3]
    )


    # 清空输入和输出
    def clear():
        input_textbox.update(value="")
        output1.update(value="")
        output2.update(value="")
        output3.update(value="")


    clear_button.click(clear, inputs=[], outputs=[])

# 启动应用
demo.launch(server_name="0.0.0.0", server_port=2223, share=True)

# # 创建 Gradio 应用
# gradio_app = gr.routes.App.create_app(demo)
#
# # 将 Gradio 应用挂载到 FastAPI 应用上
# app.mount(web_path, gradio_app)
#
# if __name__ == "__main__":
#     uvicorn.run("battle_gradio:app", host='0.0.0.0', port=2223, reload=True)
