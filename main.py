import requests
import fastapi
from fastapi import  FastAPI
import asyncio
import httpx
import datetime
from datetime import datetime, timedelta  # 날짜 자동 계산을 위해 추가
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


app = FastAPI()

# 1. 데이터 로드 및 전처리
df = pd.read_csv('Crop_recommendation.csv')

# 동시에 최대 1개까지만 요청을 보내도록 제한
sem = asyncio.Semaphore(1)

print("[P] 모델 학습을 시작합니다.")
# 머신러닝 모델 학습
try:
    # 데이터 내에 존재하는 무의미한 빈 열 자동 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s*$')]

    # 실제 파일에 존재하는 4가지 입력 특성(Feature)만 사용합니다.
    features = ['N', 'temperature', 'ph', 'rainfall']
    X = df[features]
    y = df['label']

    print("--- 사용된 데이터 변수 확인 ---")
    print(f"입력 특성: {list(X.columns)}")
    print(f"데이터 수: {df.shape[0]}개\n")

    # 전체 고유 작물 리스트를 미리 정의 (안전한 예외 처리를 위함)
    unique_labels = np.unique(y)

    # ==========================================
    # 2. 학습 데이터와 테스트 데이터 분리 (8:2)
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 3. 모델 생성 및 학습
    # ==========================================
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ==========================================
    # 4. 예측 수행
    # ==========================================
    y_pred = model.predict(X_test)

    # ==========================================
    # 5. 각 작물별 예측 정확도 계산
    # ==========================================
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    crop_accuracies = {}
    for i, label in enumerate(unique_labels):
        total_actual = cm[i].sum()  # 실제 해당 작물의 총 개수
        correct_preds = cm[i][i]  # 모델이 정확히 맞춘 개수
        accuracy_pct = (correct_preds / total_actual) * 100 if total_actual > 0 else 0
        crop_accuracies[label] = accuracy_pct

    # 결과를 데이터프레임으로 변환 후 내림차순 정렬
    accuracy_df = pd.DataFrame(list(crop_accuracies.items()), columns=['작물 (Crop)', '예측 정확도 (Accuracy %)'])
    accuracy_df = accuracy_df.sort_values(by='예측 정확도 (Accuracy %)', ascending=False).reset_index(drop=True)

    print("--- [모델 검증: 기존 데이터 기준 작물별 예측 정확도] ---")
    print(accuracy_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

    # ==========================================
    # 6. 시각화 업데이트 (Seaborn palette 경고 해결)
    # ==========================================
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # 시각화 1: 작물별 예측 정확도(%) 바 차트
    sns.barplot(
        x='예측 정확도 (Accuracy %)',
        y='작물 (Crop)',
        data=accuracy_df,
        palette='coolwarm',
        hue='작물 (Crop)',
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title('Crop Prediction Accuracy (%)', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylabel('Crop', fontsize=12)
    axes[0].set_xlim(0, 105)

    for p in axes[0].patches:
        width = p.get_width()
        axes[0].text(width + 1, p.get_y() + p.get_height() / 2 + 0.1, f'{width:.1f}%', ha="left", va="center",
                     fontsize=9)

    # 시각화 2: 4개 변수에 대한 특성 중요도 (Feature Importance)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    y_features = X.columns[indices]

    sns.barplot(
        x=importances[indices],
        y=y_features,
        palette='viridis',
        hue=y_features,
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title('Feature Importance (4 Features)', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_ylabel('Environmental Factors', fontsize=12)

    # plt.tight_layout()
    # plt.show()

    # # ==========================================
    # # 7. 가상 데이터 입력 기반 작물별 어울림 퍼센테이지 계산
    # # ==========================================
    # print("\n--- 테스트 데이터 대입 ---")
    # # 입력 순서: [N, temperature, ph, rainfall]
    # custom_data = [[90, 20.8, 6.5, 202.9]]
    #
    # # 경고 방지를 위해 DataFrame 형태로 변환
    # custom_df = pd.DataFrame(custom_data, columns=features)
    #
    # # 1) 가장 어울리는 최적의 추천 작물 단일 예측
    # predicted_crop = model.predict(custom_df)[0]
    #
    # # 2) 현재 입력된 데이터 환경이 각 작물과 얼마나 어울리는지 퍼센테이지(확률) 계산
    # pred_probabilities = model.predict_proba(custom_df)[0]
    #
    # # 3) 작물 이름과 어울림 퍼센테이지 매칭 후 데이터프레임 생성
    # match_df = pd.DataFrame({
    #     '작물 (Crop)': model.classes_,
    #     '추천 적합도 (Match %)': pred_probabilities * 100
    # })
    #
    # # 4) 가장 잘 어울리는 순서(퍼센테이지 내림차순)로 정렬
    # match_df = match_df.sort_values(by='추천 적합도 (Match %)', ascending=False).reset_index(drop=True)
    #
    # print(f"입력된 토양/기후 데이터(N, Temp, pH, Rain): {custom_data[0]}")
    # print(f"이 환경에 가장 어울리는 최적의 추천 작물은? : [{predicted_crop.upper()}] 입니다.\n")
    #
    # print("--- [입력 환경 데이터 기준 모든 작물별 어울림 퍼센테이지 상세] ---")
    # print(match_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))
except Exception as e:
    print(f"[E] 오류가 발생했습니다. {e}")


async def fetch_soil_data(lat, lon):
    global client  # 전역 HTTPX 클라이언트를 참조하기 위해 필요에 따라 유지
    async with sem:  # 세마포어 감싸기 (3개 가득 차면 나머지는 대기)
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

        # [수정] property에 phh2o(산성도)와 nitrogen(질소 함량)을 리스트로 전달합니다.
        params = {"lon": lon, "lat": lat, "property": ["phh2o", "nitrogen"]}

        try:
            response = await client.get(url, params=params, timeout=10.0)
            await asyncio.sleep(0.2)  # 요청 직후 가벼운 매너 타임

            if response.status_code == 200:
                print(f"[A] fetch_soil_data 수신됨. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                res_json = response.json()
                layers = res_json.get("properties", {}).get("layers", [])

                # 결과를 담을 딕셔너리 초기화
                soil_results = {"phh2o": None, "nitrogen": None}

                # 수신된 layers를 돌면서 각각의 값을 파싱
                for layer in layers:
                    name = layer.get("name")  # 'phh2o' 또는 'nitrogen' 반환됨
                    try:
                        # 안전하게 첫 번째 깊이(0~5cm 또는 0~30cm 등 API 기준 최고층)의 평균값을 가져옴
                        mean_val = layer["depths"][0]["values"]["mean"]
                        # SoilGrids API 특유의 10배 스케일링을 원래 단위로 복원 (/ 10)
                        soil_results[name] = mean_val / 10
                    except (KeyError, IndexError):
                        print(f"[W] {name} 데이터의 특정 깊이 값을 파싱하는 데 실패했습니다.")
                        soil_results[name] = None

                return soil_results

            if response.status_code in [429, 503]:
                print("[E] fetch_soil_data : 서버가 과부화되어 요청이 거부되었습니다.")
                return {"Error": f"서버에 대한 요청이 과부화되었습니다. 잠시후에 다시 시도해주세요. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}

        except Exception as e:
            print(f"[E] fetch_soil_data 통신 중 에러 발생: {e}")

        return None

async def get_nasa_power_data(lat, lon):
    """
    NASA POWER API를 통해 특정 위치와 기간의 기온, 강수량, 토양 수분 데이터를 가져옵니다.
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    parameters = "T2M,PRECTOTCORR,GWETROOT"

    # 2. 날짜 자동 설정 (오늘 기준)
    today = datetime.today()

    four_days_ago = today - timedelta(days=4)  # 어제
    six_days_ago = today - timedelta(days=10)  # 6일 전

    # NASA API 형식에 맞게 YYYYMMDD 문자열로 변환
    start = six_days_ago.strftime("%Y%m%d")
    end = four_days_ago.strftime("%Y%m%d")

    print(f"[P] 데이터 조회 기간: {six_days_ago.strftime('%Y-%m-%d')} ~ {four_days_ago.strftime('%Y-%m-%d')}")
    print(f"    위도: {lat}, 경도: {lon} 지역의 데이터를 NASA POWER API에서 불러오는 중...")

    query_params = {
        "parameters": parameters,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=query_params)
        response.raise_for_status()

        data = response.json()

        properties = data.get('properties', {})
        parameter_data = properties.get('parameter', {})

        temperature_dict = parameter_data.get('T2M', {})
        precipitation_dict = parameter_data.get('PRECTOTCORR', {})
        soil_moisture_dict = parameter_data.get('GWETROOT', {})

        # 전체 기간 평균값 계산
        if temperature_dict:
            avg_temp = round(sum(temperature_dict.values()) / len(temperature_dict),2)
            total_precip = round(sum(precipitation_dict.values()),2)
            avg_soil = round(sum(soil_moisture_dict.values()) / len(soil_moisture_dict),2)
            print(f"[A] 데이터 호출됨. 평균 기온 : {avg_temp}°C, 총 강수량 : {total_precip} mm, 평균 토양 수분 : {avg_soil}")

        return {
            "avg_temp":avg_temp,
            "total_precip":total_precip,
            "avg_soil":avg_soil
        }
    except requests.exceptions.RequestException as e:
        print(f"[E] API 요청 중 오류가 발생했습니다: {e}")
        return None

def crop_suitability_prediction(lat, lon):
    nasa_data = get_nasa_power_data(lat,lon)
    temp = nasa_data["avg_temp"]
    precip = nasa_data["total_precip"]

    soil_data = fetch_soil_data(lat, lon)
    phh2o = soil_data["phh2o"]
    nitrogen = soil_data["nitrogen"]

    # ==========================================
    # 7. 가상 데이터 입력 기반 작물별 어울림 퍼센테이지 계산
    # ==========================================
    print("\n--- 테스트 데이터 대입 ---")
    # 입력 순서: [N, temperature, ph, rainfall]
    custom_data = [[nitrogen, temp, phh2o, precip]]

    # 경고 방지를 위해 DataFrame 형태로 변환
    custom_df = pd.DataFrame(custom_data, columns=features)

    # 1) 가장 어울리는 최적의 추천 작물 단일 예측
    predicted_crop = model.predict(custom_df)[0]

    # 2) 현재 입력된 데이터 환경이 각 작물과 얼마나 어울리는지 퍼센테이지(확률) 계산
    pred_probabilities = model.predict_proba(custom_df)[0]

    # 3) 작물 이름과 어울림 퍼센테이지 매칭 후 데이터프레임 생성
    match_df = pd.DataFrame({
        '작물 (Crop)': model.classes_,
        '추천 적합도 (Match %)': pred_probabilities * 100
    })

    # 4) 가장 잘 어울리는 순서(퍼센테이지 내림차순)로 정렬
    match_df = match_df.sort_values(by='추천 적합도 (Match %)', ascending=False).reset_index(drop=True)

    print(f"입력된 토양/기후 데이터(N, Temp, pH, Rain): {custom_data[0]}")
    print(f"이 환경에 가장 어울리는 최적의 추천 작물은? : [{predicted_crop.upper()}] 입니다.\n")

    print("--- [입력 환경 데이터 기준 모든 작물별 어울림 퍼센테이지 상세] ---")
    print(match_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))
