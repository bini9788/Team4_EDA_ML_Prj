import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from optuna import trial

'''
    - 카테고리: 데이터 전처리
    - 개요: 무지성 인코딩
    - param: df, col_name
    - return: encoded columns dataframe or series
'''
def encode_column(df, col_name):

    # column data type이 object 또는 str 즉 범주형일 경우
    # onehot 인코딩 수행된 데이터프레임 return
    if df[col_name].dtype == object or df[col_name].dtype == str:
        onehot = OneHotEncoder(sparse=False)

        onehot_encoded_arr = onehot.fit_transform(df[col_name].values.reshape(-1, 1))
        onehot_encoded_label = onehot.categories_[0]
        onehot_encoded_df = pd.DataFrame(onehot_encoded_arr, columns=onehot_encoded_label)

        return onehot_encoded_df

    # column data type이 나머지 타입일 경우
    # 해당 컬럼의 series return
    else:
        return df[col_name]

'''
    - 카테고리: 파라미터 튜닝
    - 개요: 최적 파라미터 탐색
    - param: 
    - return: 
'''
def objective(trial):
    pass


if __name__ == '__main__':
    train_df = pd.read_csv('dataset/train_data.csv')
    print(encode_column(train_df, '건축년도'))