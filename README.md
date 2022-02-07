# DYS

2021.04. - 11.



인공지능을 활용한 맞춤 코디 어플리케이션

- 사용자가 보유한 의류를 기반으로 코디 추천
- **YOLO**를 이용하여 의류 인식 및 분류
- **의류 색 기반** 추천 조합 선택 모델 개발



# AI

사용자가 보유한 의류를 기반으로 코디를 추천해주기 위해 우리는 톤온톤, 톤인톤 2가지 코디법을 사용하였다.

- 톤온톤 : 동일한 색상 다른 톤
- 톤인톤 : 동일한 톤 다른 색상

![톤온톤-톤인톤-뜻](/img/톤온톤-톤인톤-뜻.png)



톤온톤 톤인톤 코디를 하기 위해선 톤을 알아내는 것이 필수적이었다.

그러나 색상에 대한 톤의 대략적인 지표만 있을 뿐 명확한 기준이 없었다.

톤은 PCCS 색체계를 참고하였으며, 1440개의 rgb, hsv 값을 바탕으로 톤을 분류해 놓은 데이터를 가지고 16,777,216(256\*256\*256)개의 rgb, hsv 값에 대한 톤을 분류해내는 인공지능을 제작하였다.

![퍼스널컬러인스트럭터자격증 PCCS 색체계에 대해서 : 네이버 블로그](/img/pccs.png)



앱 초기에는 사용자가 자신이 가진 옷을 찍고 상/하의, 대표색, 계절 등의 정보들을 직접 입력하도록 하였다.

그러나 이 모든 데이터를 사용자가 입력하게 할 순 없었으며, 따라서 우리는 사용자가 찍은 사진에서 상/하의를 구별해내고 대표색을 뽑아내는 작업을 수행할 필요가 있었다.

따라서 이미지 디텍션 기술에 대해 공부하였으며, 이미지 디텍션 기술 중 YOLO를 사용하였다.

또한 정확도를 높이기 위해 GrabCut과 Grad-CAM 기술에 대해 공부하였다.



## color classification

color classification은 앞서 말한 rgb, hsv 값에서 톤을 추출하는 인공지능이다.

PCCS에서는 색을 무채색과 12가지 톤으로 분류하였다.

톤을 분류하는 인공지능을 만들기 위해 1440개의 rgb와 hsv 값을 12개의 톤으로 분류한 [데이터](/color classification/data/원본/pccstorgb.xlsx)를 사용하였다.



### 모델

색을 나타내는 방법으로는 rgb, hsv 2가지 방법이 있다. 주로 rgb 값을 사용하지만 때로는 색이 hsv 값으로 들어올 수도 있기 때문에, rgb, hsv 그리고 rgb와 hsv 3가지 형식 중 어떠한 입력이 들어와도 톤을 알아낼 수 있게 모델을 구성하였다. 따라서 3가지 모델을 만들었으며, 입력층의 크기는 각각 3, 3, 6으로 설정하였다.

PCCS에 따르면 톤이 12가지이지만 무채색도 뽑아내야 하기 때문에 출력층의 크기는 13으로 설정하였다. 13가지 값이 독립적이고, 하나의 값을 선택해야 하기 때문에 출력층의 활성화 함수는 softmax를 사용하였다.

모델이 크지 않기 때문에 은닉층으로 Dense layer를 써도 충분하다고 판단하여 1024 크기의 Dense layer를 2층 쌓았으며, 활성화 함수는 GELU를 사용하였다.

데이터를 보면 무채색에 대한 데이터가 존재하지 않는다. 그러나 우리는 12가지 톤 이외에 무채색을 뽑아내야 한다. 따라서 확실한 무채색인 r=g=b 인 256개의 데이터를 추가하여 무채색을 뽑아낼 수 있게 하였다.





## image detection

이미지 디텍션 부분은 사용자의 편의를 위해 사용자가 자신이 가진 의류를 찍기만 하면 그에 대한 정보를 자동으로 입력해주는 것을 목표로 하였다.

### YOLO

이미지 딕텍션 DeepFashion2와 K-Fashion 2개의 데이터로 학습을 시도했다.

yolov5 를 사용하여 이미지 디텍션을 시도하였다.



#### 데이터 정제

DeepFashion2 데이터에서는 옷을 13가지 카테고리로 분류해 놓았다.

우리 앱에서는 상의, 하의 조합에 대한 코디만을 고려하였기에 이 중 드레스에 해당하는 데이터는 고려하지 않기로 하였다. 또한 우선 상의와 하의를 분류해내는 것이 우선이라고 판단하여 데이터를 상의와 하의 2가지 카테고리로만 분류하였다.

학습 결과 상의와 하의에 대한 분류는 높은 정확도를 보였으며, 이에 상의와 하의를 각각 기존의 카테고리대로 분류하는 작업을 하였다.



#### 결과

![val_batch0_labels](/img/val_batch0_labels.jpg)

이 사진은 DeepFashion2에서 사진에 대해 라벨링을 붙이 데이터들이며, 아래 사진은 우리가 YOLO를 사용하여 학습 시킨 모델이 사진에서 상의와 하의를 구분해낸 결과이다.

![val_batch0_pred](/img/val_batch0_pred.jpg)

완벽하게 분류를 하지는 못했지만 꽤 높은 정확도로 상의와 하의를 구분해내고 있는 것을 볼 수 있었다.

![결과](/img/결과.PNG)

YOLO 학습 결과이다.



### GrabCut 

GrabCut은 사진에서 배경을 제거해주는 기술이다.

YOLO에서 딕텍션한 부분을 GrabCut을 이용하여 배경을 제거하여 사진을 관리하고, 사진에서 의류 이외의 불필요한 정보들을 제거하기 위하여 공부를 시작하였다.

#### 결과

![image-20220206003635109](/img/image-20220206003635109.png)

위는 원본과 GrabCut을 이용하여 배경을 제거한 사진들 중 일부를 가져온 것이다.

배경이 단색이고, 유채색의 옷의 경우 분류가 잘 되지만, 배경이 복잡하고 옷이 무채색일수록 배경 제거 정확도가 떨어지는 것을 확인할 수 있었다.

그러나 GrabCut은 인물사진에서 인물 전체를 전경으로 인식하도록 하였으며, 우리는 배경 제거 이전에 YOLO를 통하여 옷의 위치한 범위를 줄일 수 있기 때문에 파라미터를 조금 수정하고 YOLO와 함께 사용한다면 유의미한 결과를 낼 수 있을 것으로 판단하였다.



# 참고 자료

- https://tuingup.tistory.com/entry/%ED%86%A4%EC%98%A8%ED%86%A4-%ED%86%A4%EC%9D%B8%ED%86%A4-%EB%9C%BB-%EC%B0%A8%EC%9D%B4-%EC%BD%94%EB%94%94%EB%B2%95-%EC%95%8C%EC%95%84%EB%B3%B4%EA%B8%B0
- https://m.blog.naver.com/koreaeducenter/221042772352



# 기타

This directory is the compressed directory of the "DYS/" directory.
Files and directories in this directory follow the rules below.

rule 1. All of the directories are copied.
rule 2. If there are more than 100 files in a directory, only 100 files of the same type are copied.
rule 3. Files over 100MB are copied only by the file name.
	(The file will contain the phrase "This file is more than 100MB.").

*** Error ***
! DYS/image detection/YOLO/datasets/DeepFashion2.zip is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/model_data/yolov3.weights is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3_1/checkpoints/yolov3_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3_1/checkpoints/yolov4_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3_1/model_data/yolov3.weights is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3_1/model_data/yolov4.weights is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp10/weights/best.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp10/weights/last.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp6/weights/best.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp6/weights/last.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp8/weights/best.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/runs/train/exp8/weights/last.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5/yolov5x.pt is more than 100MB.

! DYS/image detection/YOLO/datasets/K-Fashion/K-Fashion 이미지/Training/train_라벨링데이터.tar is more than 100MB.
! DYS/image detection/YOLO/datasets/K-Fashion/K-Fashion 이미지/Training/train_원천데이터.tar is more than 100MB.
! DYS/image detection/YOLO/datasets/K-Fashion/K-Fashion 이미지/Validation/valid_라벨링데이터.tar is more than 100MB.
! DYS/image detection/YOLO/datasets/K-Fashion/K-Fashion 이미지/Validation/valid_원천데이터.tar is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/checkpoints/yolov3_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/model_data/yolov3.weights is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/model_data/yolov4.weights is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/results/2021.09.23/checkpoints/yolov3_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/results/2021.09.24-1/checkpoints/yolov3_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3/results/2021.09.24-2/checkpoints/yolov3_custom.data-00000-of-00001 is more than 100MB.
! DYS/image detection/YOLO/TensorFlow-2.x-YOLOv3.zip is more than 100MB.
! DYS/image detection/YOLO/yolov5/yolov5x.pt is more than 100MB.
! DYS/image detection/YOLO/yolov5.zip is more than 100MB.
