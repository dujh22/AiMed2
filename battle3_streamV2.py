#!/usr/bin/env python3
# coding: utf-8

# 本脚本适用于pytho3.10及以上
# requests和json是本脚本的依赖，需要安装
import requests
import json
import streamlit as st
import time

st.set_page_config(layout="wide", page_title="模型竞技")
st.title("Med-Eval 模型竞技")

model_urls = {
    'AiMed': 'http://192.168.0.155:8000/v1/chat/completions',
    'Doctor': 'http://192.168.0.118:8088/base',
    # 'GPT': 'http://192.168.0.118:2223/v1/chat/completions',
    # 'ChatGLM': 'http://192.168.0.118:2225/v1/chat/completions',
    # 'Baichuan': 'http://192.168.0.118:2227/v1/chat/completions'
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

def main():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，这里是Med-Eval竞技场，很高兴为您服务🥰")

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
    models = model_urls.keys()  # 示例模型列表
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

        # 为每个选中的模型初始化对话历史，同时为同名模型添加序号
        model_instances = {}
        model_names = []
        for model in selected_models:
            model_instances[model] = model_instances.get(model, 0) + 1
            model_name_with_index = f"{model}_{model_instances[model]}"
            model_names.append(model_name_with_index)
            # 初始化session
            if model_name_with_index not in st.session_state:
                st.session_state[model_name_with_index] = []

        # 显示对话历史
        temp_model_name_with_index = model_names[0]
        i = -1
        for temp_message in st.session_state[temp_model_name_with_index]:
            i = i + 1
            if temp_message["role"] == "user":
                with st.chat_message("user", avatar='🧑‍💻'):
                    st.markdown(temp_message["content"])
            else:
                row_columns = []
                show_line = 3 if n > 2 else n
                for idx, model_name_with_index in enumerate(model_names):
                    if idx % show_line == 0:
                        row_columns = st.columns(show_line)
                    column_idx = idx % show_line
                    with row_columns[column_idx]:
                        st.header(model_name_with_index)
                        with st.chat_message(temp_model_name_with_index, avatar='🤖'):
                            st.markdown(st.session_state[temp_model_name_with_index][i]["content"])


        # 处理用户输入
        if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
            # 显示用户输入
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)

            # 创建模型响应的列
            row_columns = []
            show_line = 3 if n > 2 else n
            for idx, model_name_with_index in enumerate(model_names):
                if idx % show_line == 0:
                    row_columns = st.columns(show_line)
                column_idx = idx % show_line
                with row_columns[column_idx]:
                    st.header(model_name_with_index)

                    st.session_state[model_name_with_index].append({"role": "user", "content": prompt})

                    parts = model_name_with_index.rsplit('_', 1)
                    model = parts[0] if len(parts) == 2 else model_name_with_index
                    url = model_urls[model]
                    data["messages"] = st.session_state[model_name_with_index]

                    response = requests.post(url, headers=header, json=data, stream=True)
                    answer = ""

                    with st.chat_message("assistant", avatar='🤖'):
                        placeholder = st.empty()
                        # 用于避免模型无限重复
                        last_content = ""
                        num = 0
                        for chunk in response.iter_content(chunk_size=1024):
                            str_obj = chunk.decode('utf-8').replace('data: ', '').strip()
                            try:
                                json_obj = json.loads(str_obj)
                                if 'content' in json_obj['choices'][0]['delta'].keys():
                                    answer += json_obj['choices'][0]['delta']['content']
                                    if json_obj['choices'][0]['delta']['content'] == last_content:
                                        num = num + 1
                                        if num == 5:
                                            break
                                    else:
                                        last_content = json_obj['choices'][0]['delta']['content']
                                    placeholder.markdown(answer)
                            except:
                                placeholder.markdown(answer)
                    st.session_state[model_name_with_index].append({"role": "assistant", "content": answer})

                    # with st.chat_message("assistant", avatar='🤖'):
                    #     st.markdown(answer)

            # 模型评价
            models_judges = []  # 存储选择的模型

            # 单一模型评价
            models_judge = st.selectbox(f'排名第1', model_names)
            models_judges.append(models_judge)

            # 多模型评价
            # single_row_num = 3 if n > 2 else n
            # for i in range(0, n, single_row_num):
            #     cols = st.columns(single_row_num)
            #     for j in range(single_row_num):
            #         if i + j < n:
            #             with cols[j]:
            #                 # 存储每个模型选择器的值
            #                 models_judge = st.selectbox(f'排名第{i + j + 1}', model_names)
            #                 models_judges.append(models_judge)

            # st.button("清空对话", on_click=clear_chat_history)



if __name__ == "__main__":
    main()
