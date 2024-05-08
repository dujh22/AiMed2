# Med-Eval-Arena
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)

## 🔬 介绍

Med-Eval 启动命令介绍



## 😜 推理和部署

1. 启动后端基座模型

```shell
python openai_api.py
```

启动结果：

```
INFO:     Started server process [661974]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

2. 启动Arena

```shell
streamlit run battle3_streamV2.py --server.port 2222
```
启动结果：

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:2222
Network URL: http://192.168.0.155:2222
```

3. （可选）启动Arana2

```
python battle_gradio.py
```

启动结果：

```
Running on local URL:  http://0.0.0.0:2223
```



## ⚠️ 局限性

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。