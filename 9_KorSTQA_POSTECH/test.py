import json
import requests
from urllib.parse import urljoin

URL = 'http://127.0.0.1:5001/'
URL = 'http://thor.nlp.wo.tc:12349'
URL = 'http://mul.nlp.wo.tc:5000'
URL = 'http://mul.nlp.wo.tc:12349'
#URL = 'http://172.17.0.3:5000'

# test task_list
task_list_q = '/api/task_list'
response = requests.get(urljoin(URL, task_list_q))
print(response.status_code)
print(response.text)

# test task
task_q = '/api/task'
data = json.dumps(
    {
        "question": "조지 워싱턴 자신의 연설을 CD에 라이브로 녹음할 수 있었습니까?"
    }
    )
data = json.dumps(
    {
        "question": "소크라테스는 아이폰을 사용했나요?"
    }
    )
data = json.dumps(
    {
        "question": "위대한 개츠비는 소설 1984에서 영감을 얻었습니까?"
    }
    )
data = json.dumps({"question": "샤를 드골 대통령과 콘라드 아데나워 총리가 서약한 조약은 무엇인가?"})
headers = {'Content-Type': 'application/json; charset=utf-8'} # optional

response = requests.post(urljoin(URL, task_q), data=data, headers=headers)

print(response.status_code)
print(response.request)

res = response.json()
formatted_res = json.dumps(res, indent=4, ensure_ascii=False)
print(formatted_res)

print(response.raise_for_status())
