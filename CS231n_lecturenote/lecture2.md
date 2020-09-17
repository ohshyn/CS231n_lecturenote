# lecture2. Image Classfication

## Image Classification

```
컴퓨터비전 분야의 Core Task
이미지 분류?: 입력 이미지가 정해놓은 카테고리의 어디에 속하는지 고르는 일.
```

<img src="./img/lec2/image-classification.jpg" width="450px" height="300px" title="image-classification"></img>

```
고도화된 시각체계를 가진 인간에게는 쉽지만 기계에게는 정말 어려운 일.
의미론적 차이(Semantic Gap) 발생: 컴퓨터에겐 이미지는 단순히 격자형태의 숫자집합.
```

컴퓨터비전 Challenges: 동일한 물체라도 그 픽셀값들이 모조리 바뀌는 환경요소들.

```
Viewpoint variation: 시점 변화에도 동일한 물체로 인식함
Illumination: 물체가 밝은곳에 있던 어두운 곳에 있던 동일한 물체로 인식해야함.
Deformation: 물체의 형태가 변하더라도 동일한 물체로 인식해야함.
Occlusion: 물체의 일부만 볼 수 있더라도 물체로 인식해야함.
Background Clutter: 물체가 배경과 매우 유사하더라도 물체를 인식해야함.
Interclass variation: 하나의 개념(클래스)로 모든 물체의 다양한 모습을 전부 소화해 내야 한다.
```

## An image classifier

사물인식의 경우, 알고리즘 문제와 같이 직관적이고 명시적인 알고리즘은 존재하지 않는다.

```
def classify_image(image):
	# Some magic here?
	return class_label
```

- 지금까지 연구들은 사물을 인식하기위한 coded rules를 만들고자 시도해왔었다.

```
Hubel과 Wiesel의 연구 덕분에 Edges는 중요한 feature로 알려져있다.
우선 이미지에서 edges를 계산하고 다양한 corners와 egdes로 분류한다.
각 모퉁이와 엣지의 집합으로 사물을 인식하기 위한 명시적인 규칙 집합을 작성한다.
```

- 이런 접근의 문제점

```
강인하지 못하다.
확장성이 없다.
```

## Data-Driven Approach: 이 세상에 존재하는 다양한 객체들에게 유연하게 적용 가능한 확장성 있는 알고리즘

```
1. 데이터셋(이미지, 라벨)을 만든다.
2. 기계학습으로 classifier를 학습시킨다.
3. 새로운 이미지들에 대해 classifier를 평가한다.
```

```
def train(images, labels):
	# Machine Learning!
	return model
```

```
def predict(model, test_images):
	# Use model to predict labels
	return test_labels
```
