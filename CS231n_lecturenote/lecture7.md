# lecture 7. Training Neural Networks 2

## 학습목표

```
1. Fancier optimization.
2. Regularization: 네트워크의 Train/Test error 간 격차를 줄이고자 사용하는 방법.
3. Transfer learning: 원하는 양보다 더 적은 데이터를 가지고 있을 때 사용하는 방법.
```

## Fancier optimization

![optimization1](./img/lec7/optimization1.PNG)

```
Neural Net에서 가장 중요한 것은 결국 Optimization.
... 결정해둔 손실함수를 가장 작게 만드는 W를 찾는 일.
... 손실함수: W의 landscape

우측은 가장 간단한 최적화 예제.
... 2차원의 손실함수를 최소로 만드는 W(W_1, W_2)를 찾는 일.
```

- SGD: 가장 간단한 Optimization

```
우선 미니배치 안의 데이터에서 Loss 계산.
그리고 gradient의 반대 방향을 이용해서 파라미터 벡터를 업데이트.
```

![optimization2](./img/lec7/optimization1.PNG)

```
SGD의 문제점 1: 불균형한 손실함수에서 느린 업데이트.
... 불균형한 손실함수에서는 gradient의 방향이 고르지 못함.
... Loss에 영향력이 작은 차원의 가중치 업데이트는 아주 느리게, 영향력이 큰 차원은 아주 빠르게 진행됨.
... 고차원의 손실함수에서 더욱 빈번하게 발생.
```

![optimization3](./img/lec7/optimization3.PNG)

```
SGD의 문제점 2: local minima & saddle point에서 학습을 끝냄.
... SGD: gradient 계산 후 반대 방향으로 이동하는 알고리즘.
... local minima & saddle point: gradient == 0 ... 학습이 멈춤.

고차원이 될 수록 saddle point는 빈번하게 발생.
... 큰 Neural Net일 수록 local minima보다 saddle point에 취약.
... saddle point 근처의 gradient는 0에 근사하기에 학습이 느리게 진행됨 ... saddle point에 빠질 확률이 커짐.
```

![optimization4](./img/lec7/optimization4.PNG)

```
SGD의 문제점 3: gradient의 추청치를 사용하기에 noisy할 수 있다.
... 학습이 느려질 수 있음.
```

- SGD + Momentum

![optimization5](./img/lec7/optimization5.PNG)

```
단순히 velocity를 유지하여서 SGD의 모든 문제를 해결.
gradient를 계산할 때 velocity를 이용.
... rho: momentum의 비율 == velocity의 영향력 ... 보통 0.9와 같은 높은 값으로 설정.

gradient의 반대방향이 아닌 velocity의 방향으로 W가 업데이트됨.
```

![optimization6](./img/lec7/optimization6.PNG)

- Nesterov Momentum

![optimization7](./img/lec7/optimization7.PNG)

```
SGD Momentum: 가중평균(gradient, velocity)로 실제 가중치를 업데이트.

Nesterov Momentum: 벡터합(velocity, gradient)으로 실제 가중치를 업데이트.
... velocity의 방향이 잘못됐을 경우 현재 gradient의 방향을 좀더 활용할 수 있도록 개선한 것.
```

![optimization8](./img/lec7/optimization8.PNG)

```
Nesterov Momentum 계산 변형식.
... 첫 번째 수식: 기존의 momentum과 동일 ... velocity와 계산한 gradient를 일정 비율로 섞어주는 역할.
... 마지막 수식: (현재 velocity - 이전 velocity) * rho하여 더해줌 ...  현재/이전 velocity간 에러보정(error-correctin term)이 추가됨.
```

```
Q. Momentum이 Narrow한 global minima를 지나칠 수 있지 않느냐?
A. 그럴 수 있다. 하지만 그것이 Momentum의 장점이다.
... 사실 Narrow minima는 Overfit을 야기한다.
... 데이터셋이 조금만 더 모이면 일반화되어 flatten해진다는 것.
... 즉 우리가 찾는 global minima는 Flatten minima이고, 그렇기에 Momentum의 장점이 되는 것.
```

- AdaGrad

![optimization9](./img/lec7/optimization9.PNG)

```
AdaGrad: 훈련도중 계산되는 gradient를 활용하는 방법.
... W를 업데이트할 때 Update term을 gradient의 제곱항으로 나눠줌.
... 차원의 영향력(gradient)가 크면 보다 큰 수로 나눠줘서 속도를 줄임.
... 차원의 영향력(gradient)가 작으면 보다 작은수로 나눠줘서 속도를 높임.

AdaGrad의 문제점: 학습횟수가 많아지면 AdaGrad가 계속 작아짐.
... 업데이트 동안 gradient는 점점 커지고 그만큼 step size는 작아짐.
... Convex case(Flatten minima)에서는 좋은 특징.
... 하지만 Non-convex case(Saddle point 같은 경우)에서는 멈춰버릴 수 있음.
```

- RMSProp

![optimization10](./img/lec7/optimization10.PNG)

```
RMSProp: step size를 그저 누적하지 않고 (1-decay_rate)를 곱하여 AdaGrad의 문제점을 개선.
... decay_rate(보통 0.9 ~ 0.99): step의 속도를 가속/감속.
```

![optimization11](./img/lec7/optimization11.PNG)

```
Momentum 계열 vs. Grad 계열.
... Momentum은 Overshoot한 뒤에 돌아오지만 Grad는 상황에 맞게 적절히 궤적을 수정.

AdaGrad vs. RMSProp.
... AdaGrad는 일반적으로 사용되지 않음.
```

- Adam

![optimization12](./img/lec7/optimization12.PNG)

```
Adam: Momentum + RMSProp.
단, velocity가 0으로 초기화 된다면 어떤값/0에 가까운 수로 step size가 갱신되어 매우 커질 수 있음.
... Bias correction을 해줘야함.

Adam 초기 설정: 거의 모든 아키텍쳐에서 잘 동작하는 기본 설정.
... beta1 = 0.9.
... beta2 = 0.999.
... learnin_rate = 1e-3 or 5e-4
```

```
Optimization 알고리즘들의 문제점: 회전된 차원은 해결할 수 없다.
```

- Learning_rate decay

![optimization13](./img/lec7/optimization13.PNG)

```
Learning_rate: Optimization을 사용하기 위해 반드시 설정해야하는 값.
... 적절한 Learning_rate는 어떻게 구할까?

Learning_rate decay 방법.
1. 일정수준 학습된 후 Learning_rate를 낮춰서 다시 학습.
... 수렴을 잘 하고 있는 상황에서 gradient가 점점 작아지고 있을 때.
... Learnin_rate가 너무 높아 더 깊이 들어가지 못하는 상황에 사용.
2. Learning_rate decay 활용.

Adam보다 SGD Momentum을 사용할 때 자주 사용.

Leaning_rate decay 설정 순서.
... 우선 decay없이 학습을 시켜 봄.
... Loss curve를 보면서 decay가 필요한 곳이 어딘지 고려해보자.
```

- Second-Order Optimization

![optimization14](./img/lec7/optimization14.PNG)

```
지금까지의 Optimization은 전부 First-order(1차미분).
Second-Order를 사용하면 보다 효과적이다.
```

![optimization15](./img/lec7/optimization15.PNG)
![optimization16](./img/lec7/optimization16.PNG)

```
기초적인 Second-order Optimization은 learning_rate가 없지만 실제로는 필요하다.
... minima 방향으로 이동하는 것이지 minima로 가는 것이 아니기 때문.

딥러닝에서 사용할 수 없다.
... N*N의 행렬을 저장할 방법이 없다.
... 역행렬을 계산할 방법이 없다.
```

![optimization17](./img/lec7/optimization17.PNG)

```
그래서 실제로는 quasi-Newton methods를 이용.
... Full Hessian을 Low-rank approximations하는 방법.

L-BFGS 또한 Hessian을 근사하는 방법.
... 사실상 DNN에서 잘 사용하지 않는다.
... 2차근사가 stochastic case에서 잘 동작하지 않기 때문.
```

![optimization18](./img/lec7/optimization18.PNG)

```
실제로는 Adam을 가장 많이 사용.

경우에 따라 L-BFGS가 유용할 수 있음.
... full batch update가 가능: stochastic case가 아닐 때.
```

![optimization19](./img/lec7/optimization19.PNG)

```
Optimization을 잘 하여 Training error를 줄였다.
하지만 실제로 관심있는 것은 Training - Test error간 gap을 줄이는 것.
어떻게 해야할까?
```

## Model Ensembles

```
가장 쉽고 빠르게 모델의 성능을 최대화 시키는 방법.
ML 분야에서 종종 사용되는 기법.

아이디어: 모델을 하나만 학습시키는 것이 아니라 10개의 모델을 독립적으로 학습시키는 것.
결과: 10개 모델 결과의 평균.
... 모델이 늘어날 수록 Overfitting이 줄고 성능이 조금씩 향상.
... 보통 2%정도 증가.
```

- 창의적인 앙상블

![ensemble1](./img/lec7/ensemble1.PNG)

```
학습 도중 중간 모델들을 저장(snapshot)하고 앙상블로 사용.
그리고 Test time에는 여러 snapshot에서 나온 예측값들의 평균을 사용.

Learning rate를 엄청 낮췄다고 엄청 높혔다가를 반복하면서 손실함수의 다양한 지역에서 수렴할 수 있도록 할 수 있음.
```

- Smooth Ensemble: Polyak averaging

![ensemble2](./img/lec7/ensemble2.PNG)

```
Polyak averaging.
... 학습하는 동안 파라미터의 exponentially decay average를 계속 계산.
... checkpoint에서의 파라미터를 그대로 쓰지 않고 smoothly decaying average를 사용.

시도해 볼만하지만 실제로 자주 쓰이지는 않음.
```

## Regularization

