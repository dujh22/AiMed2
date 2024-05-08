# coding=utf-8
# Implements API for AiMed-13B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.


import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import re


patterns1 = r"生成处方：(.+)"
patterns2 = r"疑似诊断：(.+)"
patterns3 = r"分析报告：(.+)"

# -----------------------敏感词处理-------------------------------
class DFANode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class DFAFilter:
    def __init__(self, words):
        self.root = DFANode()
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, DFANode())
        node.is_end = True

    def filter(self, text):
        i = 0
        while i < len(text):
            node = self.root
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    return True
                j += 1
            i += 1
        return False


# 从txt文件中加载敏感词
def load_words_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        words = [line.strip() for line in file]
    return words


# 可以从多个文件中加载敏感词，并将它们全部添加到DFAFilter中
# files = [
#     "/home/djh/code/AiMed/sensitive/反动词库.txt",
#     "/home/djh/code/AiMed/sensitive/广告.txt",
#     "/home/djh/code/AiMed/sensitive/政治类.txt",
#     "/home/djh/code/AiMed/sensitive/敏感词.txt",
#     "/home/djh/code/AiMed/sensitive/敏感词及类型.txt",
#     "/home/djh/code/AiMed/sensitive/暴恐词库.txt",
#     "/home/djh/code/AiMed/sensitive/民生词库.txt",
#     "/home/djh/code/AiMed/sensitive/涉枪涉爆违法信息关键词.txt",
#     "/home/djh/code/AiMed/sensitive/色情词库.txt"
# ]  # 替换为你的txt文件路径
files = [
    "/home/djh/code/AiMed/sensitive/反动词库.txt",
    "/home/djh/code/AiMed/sensitive/政治类.txt"
]  # 替换为你的txt文件路径
all_words = []
for file_path in files:
    all_words.extend(load_words_from_txt(file_path))

dfa_filter = DFAFilter(all_words)

# 测试过滤器
# while True:
#     text = input("请输入一段文字：")
#     result = dfa_filter.filter(text)
#     print(result)

#--------------------敏感词处理结束---------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str = "Doctor"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "岳麓山数据科学与技术研究院"
    root: Optional[str] = "baichuan2-13B-Chat"
    parent: Optional[str] = "Doctor1.0"
    description: str = "本版本为Doctor基础问答模型"
    dataset: str = "Doctor(SFT DATA ONLY)"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard()
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    # query = request.messages[-1].content
    #
    # prev_messages = request.messages[:-1]
    # if len(prev_messages) > 0 and prev_messages[0].role == "system":
    #     query = prev_messages.pop(0).content + query
    #
    # history = []
    # if len(prev_messages) % 2 == 0:
    #     for i in range(0, len(prev_messages), 2):
    #         if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
    #             history.append([prev_messages[i].content, prev_messages[i+1].content])



    messages = []
    for i in request.messages:
        messages.append({"role": i.role, "content": i.content})

    sensitive_judge = dfa_filter.filter(messages[-1]["content"])
    if sensitive_judge:
        answer = "很抱歉，作为一个大语言模型，我避免回应任何包含敏感内容的询问。请您重新提问，并确保问题的表述合法且符合相关规定。我非常乐意回答您合规且有助于学习和知识分享的问题。"
        if request.stream:
            generate = predict_sensitive(messages, request.model, answer)
            return EventSourceResponse(generate, media_type="text/event-stream")
        response = answer
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop"
        )
        return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")

    # patterns1 = r"生成处方：(.+)"
    # patterns2 = r"疑似诊断：(.+)"
    # patterns3 = r"分析报告：(.+)"

    match1 = re.match(patterns1, messages[-1]["content"])
    if match1:
        messages[-1]["content"] = messages[-1]["content"][len("生成处方："):] + "请根据已经提供的信息生成一个供医疗专业人员参考的处方草案。并注意强调：生成的处方草案不能作为真实的医疗建议或指导，需由合格的医疗专业人员根据患者的具体情况进行评估和调整。"
    else:
        match2 = re.match(patterns2, messages[-1]["content"])
        if match2:
            messages[-1]["content"] = messages[-1]["content"][len("疑似诊断："):] + "请根据已经提供的病人信息和症状描述，提供一个疑似诊断作为医疗专业人员参考。并注意强调：给出的疑似诊断仅作为初步参考，不能作为最终的医疗诊断。真正的诊断需要由医疗专业人员根据病人的全面临床表现做出。"
        else:
            match3 = re.match(patterns3, messages[-1]["content"])
            if match3:
                messages[-1]["content"] = "请对如下信息进行医学解读：" + messages[-1]["content"][len("分析报告："):]

    if request.stream:
        generate = predict(messages, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response = model.chat(tokenizer, messages)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(messages: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    current_length = 0

    for new_response in model.chat(tokenizer, messages, stream=True):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

async def predict_sensitive(messages: List[List[str]], model_id: str, answer: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    # choice_data = ChatCompletionResponseStreamChoice(
    #     index=0,
    #     delta=DeltaMessage(content=answer),
    #     finish_reason=None
    # )
    # chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    # yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    for new_response in answer:

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_response),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

if __name__ == "__main__":
    model_path = "doctor_base"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
