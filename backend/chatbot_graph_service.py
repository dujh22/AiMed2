#!/usr/bin/env python3
# coding: utf-8

from question_classifier import *
from question_parser import *
from answer_search import *

from flask import Flask, request
from flask_cors import CORS
import json

rule = "我目前不知道该如何回答这个问题，您可以换一种问法，包括：乳腺癌的症状有哪些？ | 最近老流鼻涕怎么办？ | 为什么有的人会失眠？ |" \
                                 " 失眠有哪些并发症？ | 失眠的人不要吃啥？ | 耳鸣了吃点啥？ | 哪些人最好不好吃蜂蜜？ |  鹅肉有什么好处？ |  " \
                                 "肝病要吃啥药？ |  板蓝根颗粒能治啥病？ |  脑膜炎怎么才能查出来？ |全血细胞计数能查出啥来？ |  " \
                                 "怎样才能预防肾虚？ |  感冒要多久才能好？ |  高血压要怎么治？ |  白血病能治好吗？ |  什么人容易得高血压？ |  " \
                                 "糖尿病 |"

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
        User = request.form.get("user")
        ans = handler.chat_main(User)

        return_dict = {}
        if ans == "您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，希望可以帮到您。":
            return_dict['ans'] = rule
        else:
            return_dict['ans'] = ans

        return_dict = json.dumps(return_dict)

        return return_dict

@app.route('/', methods = ['GET'])
def medGPT0():
    if request.method == "GET":
        return_dict = "搭建以患者为中心的医药领域知识图谱，结合超大规模预训练语言生成模型，完成自动问答与分析服务。"

    return return_dict

if __name__ == '__main__':
    app.run(host='192.168.0.114', port = 8443, ssl_context = 'adhoc')

