# `iris detection test code`
Repository for detecting the iris circle in eye images.

### Requirements
For building and running the code you need:
- python 3.9.20
- numpy
- opencv

### Installation
```bash
$ git clone git@github.com:A-eye-LAB/a-eye-lab-research.git
$ cd iris_detection
$ pip install -r requirements.txt
```

### Usage
Run the main script with:
```
python main.py
```

#### Parameters
`min_dist` : 원 사이 최소 거리 \
`param1` : Canny 엣지 검출기 상한값 \
`param2` : 중심 검출 임계값 \
`min_radius` : 최소 반지름 (홍채 추정치에 맞게 조정) \
`max_radius` : 최대 반지름 (홍채 추정치에 맞게 조정) \
`canny_thr1` : Canny threshold1 \
`canny_thr2` : Canny threshold2 

### Output 
- Console Output : 홍채 원형 중심 좌표(x,y), 반지름, 입력 이미지 사이즈
- Image Output : `./result.png`

### Example Parameter Settings
Try the following parameter settings for the provided example images:

- `example/cataract1.jpg` 
    ```
    min_dist=1  
    param1=30  
    param2=30  
    min_radius=0 
    max_radius=45 
    canny_thr1=30
    canny_th2=100
    ```

- `example/cataract2.jpg`
    ```
    min_dist=1
    param1=30
    param2=30
    min_radius=0
    max_radius=30
    canny_thr1=10
    canny_th2=50
    ```

- `example/cataract3.jpg` & `example/cataract4.jpg`
    ```
    min_dist=1
    param1=30
    param2=30
    min_radius=0
    max_radius=150
    canny_thr1=30
    canny_th2=30
    ```

---
- 2024-11-05 yujin first created
    - 입력 이미지 크기가 일정하고, 피사체의 눈 크기(카메라와의 거리 등)가 크게 변하지 않는 경우 성능은 양호해 보이지만 추가 테스트 필요!
    - 카메라 실선 가이드에 맞춰 촬영된 데이터로 추가 테스트와 탐지 성능을 최적화하기 위한 Parameters(threshold) 조정 검토 필요!
