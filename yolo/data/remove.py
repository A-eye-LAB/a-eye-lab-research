import os
from PIL import Image

def remove_icc_profile(directory):
    # 지정된 디렉토리와 하위 디렉토리를 순회
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # 이미지 파일만 처리 (확장자를 통해 확인)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                try:
                    with Image.open(file_path) as img:
                        if img.info.get('icc_profile'):
                            # ICC 프로파일 제거를 위해 새로 저장
                            print(f"Removing ICC profile: {file_path}")
                            img.save(file_path, icc_profile=None)
                        else:
                            print(f"No ICC profile found: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

directory_path = "/workspace/data/C002" # 여기서 /path/to/your/directory를 실제 경로로 바꾸세요
remove_icc_profile(directory_path)


