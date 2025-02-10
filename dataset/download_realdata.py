import requests
import json
import os

# API 인증 토큰 발급 API : https://docs.kakaocloud.com/start/api-preparation#method-2-get-api-authentication-token
url = "https://iam.kakaocloud.com/identity/v3/auth/tokens"
data = {
    "auth": {
        "identity": {
            "methods": [
                "application_credential"
            ],
            "application_credential": {
                "name": "storage-api",
                "secret": "f341044f1f5bc96b60ff682592b7eb04aba7afae52a7d55c8542d50c6e4694c76d60ef",
                "user": {
                    "id": "87d63deb458a476892d61af4fc5c6d2c"
                }
            }
        }
    }
}

response = requests.post(url, json=data)
api_auth = response.headers.get("X-Subject-Token")

# 프로젝트 목록 조회 : Bucket ID 가져오기
url = f'https://objectstorage.kr-central-2.kakaocloud.com/v1_ext/bucket/data-collect'
headers = {
    'X-Auth-Token' : api_auth
}

response = requests.get(url, headers=headers)
body = response.json()
project_id = body['project']

# collection 에 데이터 리스트 확인
url = f'https://objectstorage.kr-central-2.kakaocloud.com/v1/{project_id}/data-collect?format=json'
headers = {
    'X-Auth-Token' : api_auth
}

response = requests.get(url, headers=headers)
body = response.json()
images = []
for i in body:
    if i['content_type'] == 'image/jpeg':
        images.append(i)

# collection 데이터 다운로드
for image in images:
    url = f'https://objectstorage.kr-central-2.kakaocloud.com/v1/{project_id}/data-collect/{image["name"]}'
    headers = {
        'X-Auth-Token' : api_auth
    }
    file_name = image['name'].split('/')[-1]
    response = requests.get(url, headers=headers)

    directory = 'data/real_data/1'
    os.makedirs(directory, exist_ok=True)

    if response.status_code == 200:
        with open(f'{os.path.join(directory,file_name)}', "wb") as f:
            f.write(response.content)
