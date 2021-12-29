
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # 모델 읽어오기


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    features = [x for x in request.form.values()]

    # 각 변수들 변환(스케일링 및 인코딩)
    encoding_and_scaling(features)

    # final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# train encoding 후 데이터 프레임 빈거 가져와서(1*406 컬럼 정보 필요하니까)
# 기본값 0으로 다 채워놨다가
# 입력값에 해당되는 컬럼만 1


# 각 변수들을 숫자로 변환
def encoding_and_scaling(features):
    # category_gu = pd.read_csv('category_gu.csv',encoding='utf-8')
    # category_dong = pd.read_csv('category_dong.csv',encoding='utf-8')
    # category_aptbrand = pd.read_csv('category_aptbrand.csv',encoding='utf-8')
    category_df = pd.read_csv('category_df.csv', encoding='utf-8')
    enc_arr = np.array([])

    # 입력값마다 for문 반복
    # for feature in features:
    for i in range(3):
        # if feature.dtype == object or feature.dtype == str:
                # # 컬럼의 유니크한 값을 리스트로 만들어둠
            col_ith_items = category_df.iloc[:, i].unique().tolist()
            onehot = OneHotEncoder()
            onehot.fit(np.array(col_ith_items).reshape(-1, 1))

             # input 값 대입
            input_ = np.array(features[i]).reshape(-1, 1)
            col_ith_enc_arr = onehot.transform(input_).toarray()
            enc_arr = np.append(enc_arr, col_ith_enc_arr)

    return enc_arr

encoding_and_scaling(['강남구', '개포동', '현대'])







if __name__ == "__main__":
    app.run(debug=True)
