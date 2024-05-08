#!/usr/bin/env python3
# coding: utf-8

from question_classifier import *
from question_parser import *
from answer_search import *

from flask import Flask, request
from flask_cors import CORS
import json
import requests
import time
from datetime import datetime

import sys
from loguru import logger
# logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题

logger.add(sink = 'log.txt', rotation="06:00")   # 创建log文件,每天6点滚动创建一次
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO") # 输出INFO级别以上的日志

logger.info("MedGPT程序启动…")

rule = "我目前不知道该如何回答这个问题，您可以换一种问法，包括：乳腺癌的症状有哪些？ | 最近老流鼻涕怎么办？ | 为什么有的人会失眠？ |" \
       " 失眠有哪些并发症？ | 失眠的人不要吃啥？ | 耳鸣了吃点啥？ | 哪些人最好不好吃蜂蜜？ |  鹅肉有什么好处？ |  " \
       "肝病要吃啥药？ |  板蓝根颗粒能治啥病？ |  脑膜炎怎么才能查出来？ |全血细胞计数能查出啥来？ |  " \
       "怎样才能预防肾虚？ |  感冒要多久才能好？ |  高血压要怎么治？ |  白血病能治好吗？ |  什么人容易得高血压？ |  " \
       "糖尿病 |"
url = 'http://192.168.0.155:7999'
url2 = 'https://43.134.20.233:8080'
header = {
    'Content-Type': 'application/json'
}
his = {}

'''问答类'''


class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，希望可以帮到您。'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return answer
        else:
            return '\n'.join(final_answers)


handler = ChatBotGraph()

app = Flask(__name__)
CORS(app, resource=r'/*')


@app.route('/chat2', methods=['POST'])
def medGPT2():
    if request.method == "POST":
        start_time5 = time.time()

        logger.info("MedGPT被调用")

        # 获取基本的请求信息
        pid = request.form.get("pid")
        logger.info("pid: {}", pid)
        User = request.form.get("user")
        logger.info("User: {}", User)
        Uid = request.form.get("uid")
        logger.info("Uid: {}", Uid)
        Sid = request.form.get("sid")
        logger.info("Sid: {}", Sid)
        His = []

        # 验证密钥，防止恶意攻击
        if pid != "$ou32nwg4er9mc-sf7vh5CN1^":
            return_dict = {'ans': "无访问权限，请联系管理员开通MedGPT权限"}
            return_dict = json.dumps(return_dict, ensure_ascii=False)
            return return_dict

        # 第一轮答案：来自知识图谱
        start_time1 = time.time()
        logger.info("KG调用请求")
        ans = handler.chat_main(User)
        logger.info("KG: {}", ans)
        end_time1 = time.time()
        print("MedGPT1.0耗时：{:.5f}秒".format(end_time1 - start_time1))

        '''
        if Uid in his:
            His = his[Uid]
        else:
            his[Uid] = []
            His = []
        print(His)
        print(type(His))
        '''
        # 第二轮答案
        data = {
            'prompt': User,
            'history': His
        }
        data2 = {
            'user': User
        }

        # 第二轮答案：来自本地大模型
        start_time2 = time.time()
        response = requests.post(url, headers=header, json=data)
        end_time2 = time.time()
        print("MedGPT2.0耗时：{:.5f}秒".format(end_time2 - start_time2))

        print(response.status_code)

        start_time4 = time.time()

        # 针对回传的答案的初步处理部分
        logger.info("ChatGLM调用请求")
        ans2 = response.json()
        logger.info("ChatGLM: {}", ans2)
        # print(ans2)
        ans3 = ans2["response"]

        # 针对模型信息的屏蔽：
        if ("模型" in ans3 or "人工智能" in ans3) and ("2022" in ans3 or "2021" in ans3):
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，训练完成于2023年5月20日，目前仍然在学习中，希望可以帮到您。'
        elif "ChatGLM-6B" in ans3 or "ChatGLM" in ans3 or "KEG" in ans3:
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，由清华大学网络大数据研究中心(DJH)开发完成，希望可以帮到您。'
        print(ans3)

        # 答案选择
        return_dict = {'ans': ans3}
        '''
        his[Uid].append(User)
        his[Uid].append(return_dict['ans'])
        '''
        return_dict = json.dumps(return_dict, ensure_ascii=False)
        logger.info("Ans: {}", return_dict)

        end_time4 = time.time()
        print("答案选择模型耗时：{:.5f}秒".format(end_time4 - start_time4))
        end_time5 = time.time()
        print("耗时：{:.5f}秒".format(end_time5 - start_time5))

        with open(file="qa.json", mode='a', encoding='utf-8', errors="ignore") as f:
            qa = {"uid": Uid, "q": User, "a": return_dict, "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            qas = json.dumps(qa, ensure_ascii=False)
            json.dump(qas, f, ensure_ascii=False)
            f.write('\n')

        return return_dict


@app.route('/chat3', methods=['POST'])
def medGPT3():
    if request.method == "POST":
        start_time5 = time.time()

        logger.info("MedGPT被调用")

        # 获取基本的请求信息
        pid = request.form.get("pid")
        logger.info("pid: {}", pid)
        User = request.form.get("user")
        logger.info("User: {}", User)
        Uid = request.form.get("uid")
        logger.info("Uid: {}", Uid)
        Sid = request.form.get("sid")
        logger.info("Sid: {}", Sid)
        His = []

        # 验证密钥，防止恶意攻击
        if pid != "$ou32nwg4er9mc-sf7vh5CN1^":
            return_dict = {'ans': "无访问权限，请联系管理员开通MedGPT权限"}
            return_dict = json.dumps(return_dict, ensure_ascii=False)
            return return_dict

        # 第一轮答案：来自知识图谱
        start_time1 = time.time()
        logger.info("KG调用请求")
        ans = handler.chat_main(User)
        logger.info("KG: {}", ans)
        end_time1 = time.time()
        print("MedGPT1.0耗时：{:.5f}秒".format(end_time1 - start_time1))

        '''
        if Uid in his:
            His = his[Uid]
        else:
            his[Uid] = []
            His = []
        print(His)
        print(type(His))
        '''
        # 第二轮答案
        data = {
            'prompt': User,
            'history': His
        }
        data2 = {
            'user': User
        }

        # 第二轮答案：来自本地大模型
        start_time2 = time.time()
        logger.info("ChatGLM调用请求")
        response = requests.post(url, headers=header, json=data)
        logger.info("ChatGLM: {}", response)
        end_time2 = time.time()
        print("MedGPT2.0耗时：{:.5f}秒".format(end_time2 - start_time2))

        # 第三轮答案：来自GPT的答案
        start_time3 = time.time()
        try:
            logger.info("ChatGPT调用请求")
            response2 = requests.post(url2, data=data2, verify=False)
            logger.info("ChatGPT: {}", response2)
        except:
            print("chatGPT宕机，请重新启动")
            response2 = response
        end_time3 = time.time()
        print("ChatGPT传回耗时：{:.5f}秒".format(end_time3 - start_time3))

        print(response.status_code)
        print(response2.status_code)

        start_time4 = time.time()

        # 针对回传的答案的初步处理部分
        ans2 = response.json()
        # print(ans2)
        ans3 = ans2["response"]
        ans33 = ""
        if response2.status_code == 200:
            ans22 = response2.json()
            print(ans22)
            ans33 = ""
            try:
                ans33 = ans22["choices"][0]["message"]["content"]
            except:
                response2.status_code = 500
        else:
            print("500----chatgpt")

        # 针对模型信息的屏蔽：
        if ("模型" in ans3 or "人工智能" in ans3) and ("2022" in ans3 or "2021" in ans3):
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，训练完成于2023年5月20日，目前仍然在学习中，希望可以帮到您。'
        elif "ChatGLM-6B" in ans3 or "ChatGLM" in ans3 or "KEG" in ans3:
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，由清华大学网络大数据研究中心(DJH)开发完成，希望可以帮到您。'
        print(ans3)

        if response2.status_code == 200:
            if ("模型" in ans33 or "人工智能" in ans33) and ("2022" in ans33 or "2021" in ans33):
                ans33 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，训练完成于2023年5月20日，目前仍然在学习中，希望可以帮到您。'
            elif "ChatGPT" in ans33 or "OpenAI" in ans33:
                ans33 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，由清华大学网络大数据研究中心(DJH)开发完成，希望可以帮到您。'
            print(ans33)

        # 答案选择
        return_dict = {'ans': ans3}
        if response2.status_code == 200:
            return_dict['ans'] = ans33
        '''
        his[Uid].append(User)
        his[Uid].append(return_dict['ans'])
        '''
        return_dict = json.dumps(return_dict, ensure_ascii=False)
        logger.info("Ans: {}", return_dict)

        end_time4 = time.time()
        print("答案选择模型耗时：{:.5f}秒".format(end_time4 - start_time4))
        end_time5 = time.time()
        print("耗时：{:.5f}秒".format(end_time5 - start_time5))

        with open(file="qa.json", mode='a', encoding='utf-8', errors="ignore") as f:
            qa = {"uid": Uid, "q": User, "a": return_dict, "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            qas = json.dumps(qa, ensure_ascii=False)
            json.dump(qas, f, ensure_ascii=False)
            f.write('\n')

        return return_dict


@app.route('/', methods=['GET'])
def medGPT0():
    if request.method == "GET":
        # return_dict = "搭建以患者为中心的医药领域知识图谱，结合超大规模预训练语言生成模型，完成自动问答与分析服务。"
        return_dict = "MedGPT正在优化升级，请耐心等待，更新周期为24h"

    return return_dict


if __name__ == '__main__':
    app.run(host='192.168.0.114', port=8440, ssl_context='adhoc')
