# AI
- Machine Learning
  - 입력 -> 학습 -> 결과
- Deep Learngin
  - 다층 신경망을 이용해 학습
- Reinforcement Learning
  - 행동을 통해 상태를 변경하여 보상을 획득

# Machine Learning 종류
- 지도학습
  - 예측(prediction - linear regression)
  - 분류(classification - logistic regression)
- 비지도학습
  - 군집(clustering)
 - 강화학습
  - 보상(reward)

# Machine Learning
- 기존 방식
  - input(x) -> function(x) -> output(y)
- 기계학습
  - training data(x,y) + learning = Model(가설)
  - test data(x) -> Model -> output(y)

# Machine Learning 과정
1. 데이터 준비
- 결측치,이상치 -> 제거 및 대치 => 전처리
2. 데이터 분할
- from sklearn.model_selection import train_test_split 
- train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=1)
- 각각의 데이터를 2차원 형태로 바꿔주는 작업 필요(reshape(-1))
3. 모델 준비
- [사용할 모델 결정](https://leedakyeong.tistory.com/entry/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1-%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D%EC%9D%B4%EB%9E%80-What-is-Logistic-Regression)
  1. prediction
    - 단순선형회귀 => from sklearn.linear_model import LinearRegression
    - 다항회귀 => from sklearn.linear_model import LinearRegression,from sklearn.preprocessing import PolynomialFeatures
    - 다항회귀이지만 모델은 단순 선형회귀 모델 (linear)를 사용한다.
      - poly = PolynomialFeatures()
      - X = poly.fit(X).transfrom(X)
      - [1,a,b,a^2,ab,b^2] X값 즉 독립변수를 원하는 차수의 다항식으로 변경해준다
        a랑 b랑 서로 연관이 있어 라고 얘기해주는 과정
        
  2. classification
  - logistic => from sklearn.linear_model import LogisticRegression
    - 합/불 True/False A/B/C/D/.. 등 분류에 사용됨
    - logistic 진행시 y(종속변수) 값은 1차원 이어야한다. 2차원으로 되어있을 시 np.ravel 함수 사용
  - [KNN(K-Nearest Neighbor)](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-6-K-%EC%B5%9C%EA%B7%BC%EC%A0%91%EC%9D%B4%EC%9B%83KNN)
    - from sklearn.neighbors import KNeighborsClassifier
    - 입력값에서 가장 가까운 k개의 데이터를 비교한다.
    - k개의 데이터 중 가장 많은 쪽의 class로 입력값을 분류한다.
    - k는 일반적으로 홀수를 사용한다(동점이 나오면 구별 어려움)
    - k는 직접 설정이 가능하며 default값은 5다.
  - [SVM](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-2%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-SVM?category=1057680)
    - from sklearn.svm import SVC
    - 모델 준비 단계에서 svc = SVC(kernel='') 커널값을 설정해야한다. linear,polynomial,rbf 등이 있다.
    - margin이 최대가 되년 결정경계(초평면) 을 정의한다.
  - [decision](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-4-%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%ACDecision-Tree?category=1057680)
    - from sklearn.tree import DecisionTreeClassifier
    - 특정한 질문에 따라 데이터를 구분하는 모델을 결정 트리 모델 이라고 한다.
    - 분기를 거듭하며 데이터를 가장 잘 구분할 수 있는 질문을 기준으로 데이터를 나눈다.
    - >> 하지만 분기를 거듭하는것이 지나치게 된다면 train과정에서는 정확하게 정보를 나눌 수 있지만 
    - >>> 오버피팅으로 인해 test과정에서의 정확도가 매우 떨어져 사용할 수 없게 된다.
    - >>>> min_sample_split 파라미터를 이용하여 한 노드가 분할사기 위한 최소 데이터 수를 제한해야 한다.
  - [randomforest](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-5-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8Random-Forest%EC%99%80-%EC%95%99%EC%83%81%EB%B8%94Ensemble?category=1057680)
    - from sklearn.ensemble import RandomForestClassifier
    - 이건 그냥 링크 들어가서 보기
  3. clustering 
    - [KMeans](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-7-K-%ED%8F%89%EA%B7%A0-%EA%B5%B0%EC%A7%91%ED%99%94-K-means-Clustering?category=1057680)
      - from sklearn.cluster import KMeans
      - 주어진 값을 통해 각 cluster의 centroid를 찾고 이를 기반으로 cluster를 구분한다.
4. 학습
5. 예측 및 평가
