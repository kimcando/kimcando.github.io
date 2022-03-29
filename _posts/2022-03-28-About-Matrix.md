---
title: About Matrix
categories:
- linear algebra; matrix, hessian
---

## 행렬 첫번째 시리즈를 소개한다.

1월부터 틈틈이 선형대수 공부를 해오다 1월초에 SVD, 얼마전 eigen decomposition과 power iteration을 정리하면서 
eigenvalue와 eigenvector에 대해 고민을 많이 했다. 
`행렬 A를 곱하더라도 방향이 바뀌지 않는 벡터는 eigenvector다. 즉, 곱하더라도 방향은 안바뀌고 원래의 방향으로 늘어나거나 줄어들기만 한다. 그 벡터에 행렬 A를 곱했을 때 scaling 되는 정도가 eigenvalue다.` 라는 정의만으로는 덜 와닿았던 것 같고,
[3Blue1Brown](https://www.youtube.com/watch?v=PFDu9oVAE-g) 영상을 보고 기하적으로 이해는 조금 더 됐는데.. 친구가 "그래서 왜 중요한데?" 라는 질문에는 결국 제대로 설명을 못하겠더라. 그때 나는 **power iteration method** 을 공부 하고 있었는데, 이걸로 eigenvector의 중요성을 설명하려다 보니 닭과 계란이 되는 느낌이었다.
그래서 양자컴퓨팅 과제하던 그 친구랑 "why eigenvector matters" 등등으로 검색하면서 행렬과 eigenvector 연관성 검색을 한참했다. 
직관적인 설명을 찾고 싶었는데 생각보다 찾긴 어려웠고 그나마 수확이라면, 엄밀하게는 수학 정의에 어긋난 부분이 있는 것 같지만 그래도 [그나마 설명을 잘한 글](https://www.dhruvonmath.com/2020/07/26/who-cares-about-eigenvectors/)을 얻었다 정도? 결론은 둘다 할 일 제대로 못하고 서브웨이 먹으러 갔다.

생각해보면 학부 때 선형대수를 별로 안좋아했는데 그건 내가 고등학교 때 기하와 벡터는 좋아했지만 행렬은 안좋아했던 이유가 크다. 
참, 거짓을 찾는 문제 유형도 별로였고 그 중의 반 정도는 반례를 찾아야하는게 썩 재미있진 않았다.
그런데 요새 행렬을 보면서 재미있다고 느끼는걸 보면 충분히 왜라는 질문 없이 받아들이기만 하고, 또 행렬이 어떻게 쓰이는지도 몰라서였던게 큰 것 같다.
여하튼 이번에는 PCA 공부하려다 공돌이 수학노트 분의 영상을 보게됐는데 꽤 재미있고, 너무 당연하게만 받아들였거나 새로운 해석이라 싶은 부분이 있어서 남겨보려한다. 
그리고 오늘 읽은 [WHEN VISION TRANSFORMERS OUTPERFORM RESNETS WITHOUT PRE-TRAINING OR STRONG DATA AUGMENTATIONS](https://arxiv.org/abs/2106.01548) 에서 어떻게 ViT, ResNet, MLP mixer의 
generalization 분석을 위해 선형대수적 특징을 가져와서 사용하는지도 짧게 남겨본다. 블로그 latex 기능을.. 안살펴봐서 수식이 어그러진 글이 될 것 같다.

(참고) **power iteration method** : 큰 차원의 행렬이 가진 모든 eigenvalue를 찾기는 컴퓨팅 측면에서 어려우니(또는 굳이 다 찾을 필요가 없으니) 가장 큰 eigenvalue를 찾는 알고리즘이다. 
임의의 벡터 `x`에 dominant eigenvalue 값이 있는 행렬 `A`(행렬에 n 개의 eigenvalue 가 있을때 압도적으로 크다고 볼 수 있는 eigenvalue)를 여러번 곱해주면 
벡터 `x`는 결국 `A`의 dominant eigenvalue에 대응되는 eigenvector 방향으로의 벡터로 변환된다. 이 성질을 이용해 반복적으로 A를 곱하고, 큰 원소값으로 normalize 해주면서 eigenvalue 찾는 법이다. 더 자세한 사항은 책이나 페이지를 찾아보자.

## 행렬의 의미

먼저 행렬에 대해 생각해보자. 크게 두가지 측면으로 보면 좋은데
1. 데이터 뭉터기 : 다차원 feature를 가진 데이터들의 집합으로 생각한다면 행렬은 데이터 뭉터기를 표현할 수 있다.
2. 선형 변환 : `kx1` 크기의 벡터에 `m x k` 행렬을 곱하는 것은 `k` 차원에 있는 벡터를 `m` 차원으로 projection 시켜주는 것인데,
  이때의 projection 의미를 벡터가 존재하는 축을 회전 또는 scaling해서 변환하는 것으로 볼 수 있다.
   
## 행렬곱의 의미

그럼 벡터에 행렬을 곱해준다는 것은 어떤 의미일까? 
1. 행렬 곱 `AB`를 생각하면 `A`의 행벡터와 `B`의 열벡터의 내적으로 연산하는 것이다. 
  보통 `벡터` 는 열벡터를 의미하고 행벡터는 열벡터에 대한 함수라고 본다. 
  다시말해서 열벡터는 `변화의 대상`이 되는 함수의 입력값이고 행벡터는 `변화를 시키는 operation` 이다.
  결국 행벡터는 열벡터를 입력으로 받아 스칼라를 출력하는 함수, functional이 된다고 볼 수 있겠다.
  * 이렇게 행벡터와 열벡터간의 `내적`을 한다는 고유의 의미를 살린 응용은 공분산 행렬에 적용된다. 공분산 행렬은 데이터 간 닮은 정도를 표현하는 행렬이다. 더 자세한 내용은 다음에 PCA를 다루면서 설명하겠다.
    
2. 행렬과 벡터의 곱은 행렬과 열벡터를 선형결합 형태로 표현하는 것이다.
    * 벡터의 선형결합이 중요한 것은 새로운 벡터 공간을 생성해줄 수 있다는 것인데, 두 개의 열벡터[1 3]^T 와 [2 4]^T 는 [3 5]^T 벡터를 포함하는 벡터 공간을 생성해낼 수 있는가? 
      만약 그렇다면 어떻게 두 벡터를 조합하면 될까? 라고 이어져서 생각할 수 있다. 
    * 이런 해석은 선형 연립 방정식의 해, 회귀 분석의 계수를 찾는 과정에서 행렬 곱을 바라보는 시선이 되겠다.
    
3. 2번의 해석을 생각해보면 행렬이 벡터의 선형 변해줬다고 볼 수 있는데 이 부분은 위의 행렬에 대한 생각 2번과 일치하기 때문에 생략한다.
    * 이런 관점으로 행렬을 보면 eigenvalue&eigenvector, PCA, SVD 등으로 응용될 수 있다.
    
특히 1번과 3번을 합쳐서 생각하면, 행렬과 벡터의 곱에서 행렬은 선형변환이라는 일종의 함수 역할을 하게 되는데 그렇다면 선형변환은 
`함수` 가 가지는 정의를 만족하는가?에 대한 고민이 좀더 필요하다. 

wiki에서의 함수 정의는 `In mathematics, a function from a set X to a set Y assigns to each element of X exactly one element of Y` 로 
정의역의 각 원소를 정확히 하나의 공역 원소에 대응시키는 것이다. f: X -> Y 라는 notation으로보면 정의역 X와 공역 Y가 있을 때, 정의역 X는 함수 f에 의해서 공역 중 하나로 대응되는 치역 f(x)을 만드는 것이다. 
결국 선형변환을 함수로 보려면 선형변환의 정의역, 공역, 치역은 어느것이고 공역 중 하나로 대응될 수 있는가? 에 대한 해결을 하면 된다. 여기서 Gilbert 교수님의 유명한 row space, null space, column space, left null space 가 나오는데 정확히 이해 못한 부분이 있어서 다음 글로 넘긴다.(^^)
다만 행렬이 선형함수이고, 자세하게는 행렬을 구성하는 행벡터가 선형함수는라는 것. 이 함수는 벡터의 특성을 가지기에 행벡터도 마찬가지로 벡터다!(행벡터인데 벡터가 아닌게 더 이상하지 않는가?)를 정확히 이해해야 위의 4개의 공간, 쌍대 공간으로 편하게 넘어갈 수 있는 것 같으니 이후에 한번 더 읽어봐야겠다.


## Hessian matrix 
[Hessian matrix](https://seongkyun.github.io/study/2019/03/18/Hessian_matrix/) 는 scalar-value를 갖는 다변수 함수의 2계 도함수(second-order)를 이용하여 만든 행렬이다. 위에 주르륵 적은 행렬은 `데이터` 자체 혹은 `함수` 로 보는데, 
hessian 은 어떤 함수가 가지는 방향을 표현하는 행렬이다. (갑자기 어떤 함수의 2계 도함수를 설명하는 것이라니.. 좀 어색할 수 있다.) 어쨌던 이 행렬은 2계 도함수로 표현된 행렬이기 때문에 정방행렬이고 symmetric 하다. 
왜냐하면 함수가 f(x_1, x_2)로 정의될 때 \partial x_1, x_2를 구할 때 x_1으로 미분하고 x_2로 미분하나, x_2로 미분하고 x_1으로 하나 순서는 상관없다. 그리고 이렇게 2번 미분한걸 행렬로 표현하는거라 행과 열의 크기가 같기 때문이다.

보통 square & symmetric 행렬 `A`의 eigenvalue 값이 모두 양수면 positive-definite이라고 한다. positive definite의 정의는 [여기](https://pasus.tistory.com/10) 를 참고하되, 
이게 중요한 이유는 함수가 볼록함수(convex)이냐 오목함수(concave)이냐를 판단하는 중요한 요소이기 때문이다.
간단하게만 설명하자면 positive definite의 정의인 `x^TAx>0` 는 `x^T(Ax)>0` 이고, 이것은 벡터 `x`에 `A`를 곱한 결과 벡터와 `x^T`를 곱했을 때 0보다 커야한다는 것이다. 
만약 `A`가 `x`를 변환할 때 90도 이상의 변환을 주지않는다면 `Ax`와 `x^T`를 내적하면 늘 양수값이 나오게 될거다. 늘 양수값이 나오는 함수가 뭐가 있을까? 간단하게 x^2+1를 생각할 수 있고, 예전에 배운 아래로 볼록 함수를 생각하면 된다! 
볼록함수는 global minimum을 갖게 되는데, 양의 실수값만 가지는 2차 함수는 양의 무한 값을 가지지만 가장 작은 함수값을 가지는 점이 있지 않은가? 그런 꼴을 떠올리면 된다.

## Eigenvalue & Eigenvector
eigenvalue와 eigenvector의 정의는 위에서 설명했지만 wikipedia와 linear algebra and its application 정의를 각각 보자.

>In linear algebra, an eigenvector (/ˈaɪɡənˌvɛktər/) or characteristic vector of a linear transformation is a nonzero vector that changes at most by a scalar factor when that linear transformation is applied to it. The corresponding eigenvalue, often denoted by {\displaystyle \lambda }\lambda , is the factor by which the eigenvector is scaled.

> An eigenvector of an n by n matrix A is a nonzero vector x such that Ax = \lambda x for some scalar \lambda. A scalar \lambda is called an eigenvalue of A if there is a nontrivial solution x of Ax = \lambda x; such an x is called an eigenvector corresponding to \lambda

그러니까 벡터에 행렬을 곱해줬는데 이 벡터가 방향은 안바뀌고 크기만 바뀌더라. 라는 소리다.

## Hessian & Eigenvalue

그럼 eigenvalue가 큰 행렬을 갖는다는 것은 뭘까? 어떤 행렬을 곱해줬을 때 방향은 안변할지라도 그 방향으로 변화가 크게 일어날 것이라는 말이다.
반면에 eigenvalue가 작다는 것은 그 방향으로의 변화가 굉장히 작다는 말이 된다. Hessian 행렬이 어떤 함수의 변화도를 나타내는 행렬이라는걸 생각하면, 
이 함수는 어떤 방향으로는 굉장히 가파르게 바뀌고 어떤 방향으로는 완만하게 변화한다고 생각할 수 있다.

오늘 리뷰를 한 논문 "WHEN VISION TRANSFORMERS OUTPERFORM RESNETS WITHOUT PRE-TRAINING OR STRONG DATA AUGMENTATIONS" 은 
ViT, MLP mixer 네트워크가 왜 많은 데이터를 통한 pre-training 또는 strong data augmentation을 주지 않고서는 generalization 성능이 ResNet보다 낮은지를 
loss landscape으로 설명한다. [Loss landscape visualization](https://arxiv.org/abs/1712.09913) 논문에서 사용한 방식으로 시각화해서 보여주지만 이에 더불어 training loss의 
2차 미분 정보를 담은 hessian 행렬의 max eigenvalue 값도 사용한다. 논문에서는 loss landscape 자체도 ViT, MLP mixer가 더 날카롭고, hessian의 eigenvalue 도 resnet보다 
훨씬 큰 것을 미루어보다 함수의 형태(curvature)가 굉장히 날카로울 수 있다고 설명한다. 
따라서 non-convex 한 딥러닝 모델에서 특히 sharp한 loss land scape을 가지는 경우 training loss 만을 줄이는 것이 아니라 smooth한 loss landscape을 가질 수 있도록 최적화를 고려해야하며, 
sharp aware minimization (SAM) optimizer를 사용해 pre-training, strong data augmentation 없이도 비슷한 파라미터 규모의 resnet과 비교해 높은 성능을 낼 수 있음을 보였다. 
물론 그 이외에 초반 레이어를 sparse하게 만들어 pruning의 가능성을 높일 수 있고, attention map도 더 유의미하게 나온다는 장점도 보였으나 이는 생략한다. 
이 논문 말고도 네이버에서 낸 [How do vision transformers work](https://arxiv.org/abs/2202.06709) 에서 마찬가지로 hessian으로 분석을 하는데, 
ViT와 resnet의 loss landscape 경향성이 좀 달라서 확인해봐야할 것 같다.

## 끝으로
글을 작성하며 러프하게 적은 부분도 있고, 헷갈리고 오류가 있는 부분도 있겠으나 오랜만에 공부하고 이해한 내용으로 정보성 글을 적으니 힘들지만 보람차긴하다. 
오늘은 Hessian의 eigenvalue에 꽂혀서 PCA도 정리하려했는데, 정확히 어떻게 구현되는지 보려면 좀더 시간을 들여야겠다.
Loss landscape 내용이나 second-order matrix는 한 2년전부터 코드 봐야지..봐야지.. 한거였는데, 조만간 시간 내어 코드 리뷰해봐야겠다.(제발)
끝으로 시간 관계상 퇴고를 못했는데, 퇴고하면서 그림도 좀 끼워넣고 수식도 제대로 넣어야지 ~_~ 졸린다.



## 참고
* [행렬곱에 대한 새로운 시각](https://www.youtube.com/watch?v=X564toU2riA)
* [행벡터의 의미와 벡터의 내적](https://www.youtube.com/watch?v=ZH79kAgC3I4&list=PL5yujGYFVt0BCu7DXfEgD7M51Tj6S7s4A&index=4)
* [4개 주요 부분 공간 간의 관계](https://www.youtube.com/watch?v=VYKbaSoj_Z4&list=PL5yujGYFVt0BCu7DXfEgD7M51Tj6S7s4A&index=11)
* [positive definite](https://angeloyeo.github.io/2021/12/20/positive_definite.html)