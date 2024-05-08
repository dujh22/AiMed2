import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 MedGPT 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 MedGPT 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 MedGPT 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        response = ""
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    build_prompt(history)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        build_prompt(history)
        if "ChatGLM-6B" in response:
            response = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，希望可以帮到您。'
        elif "KEG" in response:
            response = '清华大学网络大数据研究中心'
        print(response)

if __name__ == "__main__":
    main()
