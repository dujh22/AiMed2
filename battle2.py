#!/usr/bin/env python3
# coding: utf-8

# 本脚本适用于pytho3.10及以上
# requests和json是本脚本的依赖，需要安装
import requests
import json
import streamlit as st
import concurrent.futures

st.set_page_config(page_title="模型竞技")
st.title("Med-Eval 模型竞技")

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

# 清空对话
def clear_chat_history():
    del st.session_state.messages

# 初始化历史信息显示
def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，这里是Med-Eval竞技场，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

def fetch_response(model_url, data, placeholder):
    response = requests.post(model_url, headers=header, json=data, stream=True)
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            str_obj = chunk.decode('utf-8').replace('data: ', '').strip()
            json_obj = json.loads(str_obj)
            if 'content' in json_obj['choices'][0]['delta'].keys():
                answer = json_obj['choices'][0]['delta']['content']
                placeholder.markdown(answer)


# 为每个模型创建一个线程
def start_model_threads(data, selected_models):
    placeholders = {}
    for i, model in enumerate(selected_models):
        cols = st.columns(2)
        with cols[i % 2]:
            placeholders[model] = st.empty()
            st.write(f"模型 {model} 的响应:")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for model in selected_models:
            executor.submit(fetch_response, model_urls[model], data, placeholders[model])


def main():

    # 使用columns来并排显示
    col1, col2 = st.columns(2)
    # 在第一个列中设置竞技模型数量
    with col1:
        # 设置竞技模型数量
        n = st.number_input('模型数量', min_value=1, max_value=12, value=3)
    # 在第二个列中选择竞技项目
    with col2:
        # 竞技项目选择
        competiton_types = ['用户交互', 'N+医疗场景', 'Med-Eval测评', '其它测评']
        competition_type = st.selectbox('竞技项目', competiton_types)

    # 竞技模型选择
    models = ['AiMed', 'Doctor', 'GPT4']  # 示例模型列表
    selected_models = []  # 存储选择的模型

    single_row_num = 3 if n > 2 else n
    for i in range(0, n, single_row_num):
        cols = st.columns(single_row_num)
        for j in range(single_row_num):
            if i + j < n:
                with cols[j]:
                    # 存储每个模型选择器的值
                    selected_model = st.selectbox(f'选择模型{i + j + 1}', models)
                    selected_models.append(selected_model)

    if competition_type == '用户交互':

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})
            # print(f"[user] {prompt}", flush=True)

            data["messages"] = messages
            start_model_threads(data, selected_models)


            # messages.append({"role": "assistant", "content": answer})
            #
            # # 竞技结果评价
            # # 创建三个并排的列
            # clear, good, bad = st.columns(3)
            # with good:
            #     if st.button('好', key='good', help='点击此按钮表示好的评价'):
            #         st.write('用户评价：好')
            # with bad:
            #     if st.button('坏', key='bad', help='点击此按钮表示坏的评价'):
            #         st.write('用户评价：坏')
            # with clear:
            #     st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
