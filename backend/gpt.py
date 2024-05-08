import requests

#url = 'http://192.168.0.155:7999'
url = 'http://43.134.20.233:8080'
header = {
    'Content-Type': 'application/json'
}
inpu = "你好"
data = {
    'prompt': inpu,
    'history': []
}
response = requests.post(url, headers=header, json=data)
print(response.status_code)
# print(response.json())
ans = response.json()
ans2 = ans["response"]
if "ChatGLM-6B" in ans2:
    ans2 = '您好，我是清华大学网络大数据研究中心的医药智能助理MedGPT，希望可以帮到您。'
elif "KEG" in ans2:
    ans2 = '清华大学网络大数据研究中心'
print(ans2)
