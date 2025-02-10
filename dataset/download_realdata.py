import requests
import json
import os
import shutil
import gdown
import zipfile

def download_kakaocloud(save_directory):
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

        directory = os.path.join(save_directory, '1')
        os.makedirs(directory, exist_ok=True)

        if response.status_code == 200:
            with open(f'{os.path.join(directory,file_name)}', "wb") as f:
                f.write(response.content)

def custom_sort(filename):
    parts = filename.split('_')

    # 날짜 추출
    date_part = parts[0] if len(parts) > 0 else ""

    # 숫자 부분 추출 (숫자가 아닌 경우 예외처리)
    try:
        num_part = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else float('inf')
    except ValueError:
        num_part = float('inf')  # 숫자가 없으면 가장 뒤로 정렬

    return (date_part, num_part)
def download_google(save_directory):

    # Google cloud 에서 데이터 가져오기
    file_id = "1XIiMVZIzllpHguB8dv7g0q3lq_UyI5KB"
    output = "download_file.zip"

    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

    # ZIP 파일 압축 해제
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("dataset")  # 압축 해제할 폴더 지정

    # 다운로드한 ZIP 파일 삭제 (선택 사항)
    os.remove(output)

    root_dir = "./dataset"
    count = 0
    two_count = 0

    directory = os.path.join(save_directory, '0')
    os.makedirs(directory, exist_ok=True)

    for subdir, _, files in os.walk(root_dir):
        files.sort(key=custom_sort)
        for file in files:
            file_name = file.split('.')[0]
            extection = file.split('.')[1]
            if extection == "DS_Store": continue
            date = file_name.split('_')[0]
            direction = file_name.split('_')[2]

            origin_filepath = os.path.join(subdir, file)
            new_filename = f'{count:02}_{date}_{direction}.{extection}'
            new_filepath = os.path.join(save_directory,"0", new_filename)

            shutil.move(origin_filepath, new_filepath)

            two_count += 1
            if two_count == 2:
                count += 1
                two_count = 0


if __name__ == '__main__':

    save_directory = './data/real_data'
    download_kakaocloud(save_directory)
    download_google(save_directory)


