import requests

url = 'http://10.0.0.29:8443'
header = {
    'Content-Type': 'application/json'
}
inpu = "你好"
data = {
    'prompt': inpu,
    'history': []
}
response = requests.post(url, '/getSymptomCategory')
print(response.status_code)
# print(response.json())
ans = response.json()
ans2 = ans["response"]
print(ans2)
