import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from optuna import Trial
import optuna
from optuna.samplers import TPESampler

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
    rmse 값 도출
'''
def RMSE(y, y_pred):
    rmse = mean_squared_error(y, y_pred) ** 0.5
    return rmse


'''
    - 카테고리: 모델링
    - 개요: 머신러닝 모델링 수행 및 점수 도출
        - 교차 검증 방법으로 TimeSeriesSplit 수행
    - param: 

        1. model_tuple => ex. ('LR', LinearRegression())
        2. X_train, y_train, X_test, y_test

    - return: rmse
'''
def execute_modeling(model_tuple, X_train, y_train, X_test, y_test):

    name = model_tuple[0]
    model = model_tuple[1]

    # 각 모델에 대하여 실질적 학습 수행
    clf = model.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # 각 모델의 rmse 점수 도출 
    rmse = RMSE(y_test, pred)
    print(f'{name} rmse: {rmse}')

    # TimeSeries Cross validation 
    tscv = TimeSeriesSplit(n_splits=15)

    # 각 모델에 대하여 교차 검증한 결과 점수 확인
    # scoring parameter option 어캐 줘야 함?
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_results = np.sqrt(-cv_results)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    return rmse



if __name__ == '__main__':

    from sklearn.linear_model import LinearRegression 
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import lightgbm as lgb

    # 1. test encode_column function
    train_df = pd.read_csv('dataset/train_data.csv')
    # print(encode_column(train_df, '건축년도'))


    # 2. test execute_modeling function
    preprocessed_train_df = pd.read_csv('dataset/standard_dataset.csv').drop('Unnamed: 0', axis=1)
    preprocessed_train_df = preprocessed_train_df.drop_duplicates()

    X_train = preprocessed_train_df[preprocessed_train_df['transaction_year_month_ec'] < 30].drop(['price_log'], axis=1) 
    y_train = preprocessed_train_df[preprocessed_train_df['transaction_year_month_ec'] < 30]['price_log'] 
    X_test = preprocessed_train_df[preprocessed_train_df['transaction_year_month_ec'] >= 30].drop(['price_log'], axis=1)
    y_test = preprocessed_train_df[preprocessed_train_df['transaction_year_month_ec'] >= 30]['price_log']
    
    model_list = [
                    ('LR', LinearRegression()), 
                    ('RF', RandomForestRegressor(n_estimators = 10)),
                    ('model_xgb', xgb.XGBRegressor(n_estimators=500, max_depth=9, min_child_weight=5, gamma=0.1, n_jobs=-1)),
                    ('model_lgb', lgb.LGBMRegressor(n_estimators=500, max_depth=9, min_child_weight=5, n_jobs=-1))
                ]

    for model_tuple in model_list:
        # print(execute_modeling(model_tuple, X_train, y_train, X_test, y_test))
        pass


    # 3. test get_best_param function
    def object(trial:Trial, X_train, y_train, X_test, y_test):
        params = {
            "n_estimators" : trial.suggest_int('n_estimators', 500, 4000),
            'max_depth':trial.suggest_int('max_depth', 8, 16),
            'min_child_weight':trial.suggest_int('min_child_weight', 1, 300),
            'gamma':trial.suggest_int('gamma', 1, 3),
            'learning_rate': 0.01,
            'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
            'nthread' : -1,
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0] ),
            'random_state': 42
        }
        
        test_model = xgb.XGBRegressor(**params)
        test_model_score = execute_modeling(('XGBR', test_model), X_train, y_train, X_test, y_test)

        return test_model_score

    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial: object(trial, X_train, y_train, X_test, y_test), n_trials=10)

    best_score = study.best_value
    best_param_dict = study.best_trial.params

    # print(best_score, best_param_dict)

    # 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
    optuna.visualization.plot_param_importances(study)

    # 하이퍼파라미터 최적화 과정을 확인
    optuna.visualization.plot_optimization_history(study)