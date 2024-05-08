#!/usr/bin/env python3
# coding: utf-8

from question_classifier import *
from question_parser import *
from answer_search import *

from flask import Flask, request
from flask_cors import CORS
import json
import requests

rule = "我目前不知道该如何回答这个问题，您可以换一种问法，包括：乳腺癌的症状有哪些？ | 最近老流鼻涕怎么办？ | 为什么有的人会失眠？ |" \
                                 " 失眠有哪些并发症？ | 失眠的人不要吃啥？ | 耳鸣了吃点啥？ | 哪些人最好不好吃蜂蜜？ |  鹅肉有什么好处？ |  " \
                                 "肝病要吃啥药？ |  板蓝根颗粒能治啥病？ |  脑膜炎怎么才能查出来？ |全血细胞计数能查出啥来？ |  " \
                                 "怎样才能预防肾虚？ |  感冒要多久才能好？ |  高血压要怎么治？ |  白血病能治好吗？ |  什么人容易得高血压？ |  " \
                                 "糖尿病 |"
url = 'http://192.168.0.155:7999'
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
CORS(app, resource = r'/*')

@app.route('/chat2', methods = ['POST'])
def medGPT2():
    if request.method == "POST":
        # 第一轮答案
        User = request.form.get("user")
        ans = handler.chat_main(User)
        Uid = request.form.get("uid")
        His = []
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
        response = requests.post(url, headers=header, json=data)
        print(response.status_code)
        ans2 = response.json()
        # print(ans2)
        ans3 = ans2["response"]
        if ("模型" in ans3 or "人工智能" in ans3) and ("2022" in ans3 or "2021" in ans3):
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，训练完成于2023年5月20日，目前仍然在学习中，希望可以帮到您。'
        elif "ChatGLM-6B" in ans3 or "ChatGLM" in ans3 or "KEG" in ans3:
            ans3 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，由清华大学网络大数据研究中心(DJH)开发完成，希望可以帮到您。'

        print(ans3)

        return_dict = {'ans': ans3}
        if ans == "您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，希望可以帮到您。":
            if "不知道" in ans:
                return_dict['ans'] = rule
            else:
                return_dict['ans'] = ans3
        else:
            return_dict['ans'] = ans
        '''
        his[Uid].append(User)
        his[Uid].append(return_dict['ans'])
        '''
        return_dict = json.dumps(return_dict)

        return return_dict

@app.route('/', methods = ['GET'])
def medGPT0():
    if request.method == "GET":
        return_dict = "搭建以患者为中心的医药领域知识图谱，结合超大规模预训练语言生成模型，完成自动问答与分析服务。"

    return return_dict

if __name__ == '__main__':
    app.run(host='192.168.0.114', port = 8440, ssl_context = 'adhoc')

