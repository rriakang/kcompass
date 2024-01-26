import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # F 모듈 임포트 추가
import pandas as pd
import pymysql
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# MySQL 연결
db = pymysql.connect(host='127.0.0.1', port=3305, user='root', passwd='Xptmxm1212!@', db='admin', charset='utf8')

# SQL 쿼리를 사용하여 데이터 가져오기
sql_query = 'SELECT * FROM test_1;'
data = pd.read_sql(sql_query, db)

# 연결 종료
db.close()

# NaN 값을 KNN 알고리즘으로 대체
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 데이터 표준화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed.drop('Initial_AUDIT', axis=1))
data_filled_scaled = pd.DataFrame(data_scaled, columns=data_imputed.columns[1:])
data_filled_scaled['Initial_AUDIT'] = data_imputed['Initial_AUDIT']

# 예측할 데이터
y = torch.FloatTensor(data_filled_scaled['Initial_AUDIT'].values)

# 예측에 사용할 특성 데이터
X = torch.FloatTensor(data_filled_scaled.drop('Initial_AUDIT', axis=1).values)

# 모델 클래스 정의 (선형 회귀)
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 생성
print(X.shape[1])
model = LinearRegression(input_dim=X.shape[1])

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    hypothesis = model(X)

    # cost 계산 (평균 제곱 오차 손실 함수 사용)
    cost = F.mse_loss(hypothesis.view(-1), y)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 20 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


# 모델 저장
torch.save(model.state_dict(), 'linear_model.pth')
