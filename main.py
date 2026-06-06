import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, Request
import httpx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# API 호출 속도를 제어하기 위한 세마포어 (서버 차단 방지)
sem = asyncio.Semaphore(1)

# ==========================================
# 1. 데이터 로드 및 모델 사전 학습 (서버 시작 시 1회만 실행)
# ==========================================
print("[P] 모델 학습을 시작합니다.")
try:
    df = pd.read_csv('Crop_recommendation.csv')

    # 데이터 내에 존재하는 무의미한 빈 열 자동 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed|^\s*$')]

    # 특성(Feature)과 레이블 정의
    features = ['N', 'temperature', 'ph', 'rainfall']
    X = df[features]
    y = df['label']

    print("--- 사용된 데이터 변수 확인 ---")
    print(f"입력 특성: {list(X.columns)}")
    print(f"데이터 수: {df.shape[0]}개\n")

    unique_labels = np.unique(y)

    # 학습 데이터와 테스트 데이터 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 생성 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 각 작물별 예측 정확도 계산
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    crop_accuracies = {}
    for i, label in enumerate(unique_labels):
        total_actual = cm[i].sum()
        correct_preds = cm[i][i]
        accuracy_pct = (correct_preds / total_actual) * 100 if total_actual > 0 else 0
        crop_accuracies[label] = accuracy_pct

    accuracy_df = pd.DataFrame(list(crop_accuracies.items()), columns=['작물 (Crop)', '예측 정확도 (Accuracy %)'])
    accuracy_df = accuracy_df.sort_values(by='예측 정확도 (Accuracy %)', ascending=False).reset_index(drop=True)

    print("--- [모델 검증: 기존 데이터 기준 작물별 예측 정확도] ---")
    print(accuracy_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

except Exception as e:
    print(f"[E] 초기 모델 학습 중 치명적인 오류가 발생했습니다: {e}")


# ==========================================
# 2. FastAPI 최신 표준 생명주기 관리 (Lifespan 패턴 적용)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 전역 변수 대신 app.state 구조에 비동기 클라이언트를 안전하게 격리 바인딩
    app.state.client = httpx.AsyncClient()
    print("[P] FastAPI State 내부에 비동기 httpx.AsyncClient가 생성되었습니다. (Lifespan)")

    yield  # 이 시점에서 FastAPI 서버가 클라이언트 요청을 대기하기 시작함

    # [Shutdown] 서버가 꺼질 때 클라이언트 자원을 완전히 반환
    await app.state.client.aclose()
    print("[P] 비동기 httpx.AsyncClient 자원이 안전하게 해제되었습니다. (Lifespan)")


# Lifespan 핸들러를 등록하여 FastAPI 인스턴스 생성
app = FastAPI(lifespan=lifespan)


# ==========================================
# 3. 비동기 외부 API 연동 함수 정의 (의존성 주입)
# ==========================================
async def fetch_soil_data(client: httpx.AsyncClient, lat, lon):
    """
    인자로 주입된 httpx.AsyncClient를 사용하여
    ISRIC SoilGrids API를 통해 토양 산도(pH)와 질소 함량(nitrogen)을 가져옵니다.
    """
    async with sem:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {"lon": lon, "lat": lat, "property": ["phh2o", "nitrogen"]}

        try:
            response = await client.get(url, params=params, timeout=30.0)
            await asyncio.sleep(0.2)  # 요청 직후 매너 타임

            if response.status_code == 200:
                print(f"[A] fetch_soil_data 수신됨. {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                res_json = response.json()
                layers = res_json["properties"]["layers"]

                # API 응답 구조 레이어 인덱스로 직접 타겟팅 (순서: phh2o, nitrogen)
                ph_val = layers[0]["depths"][0]["values"]["mean"] / 10
                n_val = layers[1]["depths"][0]["values"]["mean"] / 10

                return {"phh2o": ph_val, "nitrogen": n_val}

            print(f"[E] fetch_soil_data 에러 상태코드: {response.status_code}")
            return None

        except Exception as e:
            print(f"[E] fetch_soil_data 내부 예외 발생: {type(e).__name__} - {e}")
            return None


async def get_nasa_power_data(client: httpx.AsyncClient, lat, lon):
    """
    인자로 주입된 httpx.AsyncClient를 사용하여
    NASA POWER API로부터 평균 기온과 총 강수량을 가져옵니다.
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    parameters = "T2M,PRECTOTCORR"

    today = datetime.today()
    four_days_ago = today - timedelta(days=4)
    six_days_ago = today - timedelta(days=10)

    start = six_days_ago.strftime("%Y%m%d")
    end = four_days_ago.strftime("%Y%m%d")

    print(f"[P] NASA 데이터 조회 기간: {six_days_ago.strftime('%Y-%m-%d')} ~ {four_days_ago.strftime('%Y-%m-%d')}")

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
        response = await client.get(url, params=query_params, timeout=15.0)
        response.raise_for_status()

        data = response.json()
        parameter_data = data.get('properties', {}).get('parameter', {})

        temperature_dict = parameter_data.get('T2M', {})
        precipitation_dict = parameter_data.get('PRECTOTCORR', {})

        if temperature_dict:
            avg_temp = round(sum(temperature_dict.values()) / len(temperature_dict), 2)
            total_precip = round(sum(precipitation_dict.values()), 2)
            print(f"[A] NASA 호출 완료. 평균 기온: {avg_temp}°C, 총 강수량: {total_precip} mm")

            return {
                "avg_temp": avg_temp,
                "total_precip": total_precip
            }
    except Exception as e:
        print(f"[E] NASA API 요청 중 오류 발생: {e}")

    return None


# ==========================================
# 4. FastAPI 비동기 추천 라우터
# ==========================================
@app.get("/predict/{lat}/{lon}")
async def crop_suitability_prediction(request: Request, lat: float, lon: float):
    """
    요청(request) 객체에서 lifespan 인스턴스를 추출하여 하부 API 함수에 명시적으로 인자를 주입합니다.
    이를 통해 스코프 불일치로 인한 None 인식 버그를 물리적으로 제거합니다.
    """
    # lifespan 과정에서 앱 자체에 등록해 둔 비동기 httpx 클라이언트 추출
    api_client = request.app.state.client

    # 두 외부 API 요청에 안전한 컨텍스트 인스턴스를 바인딩하고 병렬 스케줄링 실행
    nasa_task = get_nasa_power_data(api_client, lat, lon)
    soil_task = fetch_soil_data(api_client, lat, lon)

    nasa_data, soil_data = await asyncio.gather(nasa_task, soil_task)

    # 데이터 누락 예외 처리
    if not nasa_data or not soil_data:
        return {"status": "error", "message": "외부 기후 및 토양 데이터를 통합 수집하지 못했습니다. 서버 콘솔 창의 에러 원인을 파악하세요."}

    temp = nasa_data.get("avg_temp")
    precip = nasa_data.get("total_precip") * 10  # ML 데이터셋 스케일에 최적화하도록 가중치 보정
    phh2o = soil_data.get("phh2o")
    nitrogen = soil_data.get("nitrogen")

    if None in [temp, precip, phh2o, nitrogen]:
        return {"status": "error", "message": "외부 API로부터 유효한 인자 환경 성분을 채우지 못해 계산을 중단합니다."}

    # 머신러닝 예측 파이프라인
    print("\n--- 실시간 수집 데이터 기반 최종 머신러닝 추론 시작 ---")
    custom_data = [[nitrogen, temp, phh2o, precip]]
    custom_df = pd.DataFrame(custom_data, columns=features)

    predicted_crop = model.predict(custom_df)[0]
    pred_probabilities = model.predict_proba(custom_df)[0]

    # 작물별 추천 매칭 세부율 정렬
    match_df = pd.DataFrame({
        '작물 (Crop)': model.classes_,
        '추천 적합도 (Match %)': np.round(pred_probabilities * 100, 2)
    })
    match_df = match_df.sort_values(by='추천 적합도 (Match %)', ascending=False).reset_index(drop=True)

    # 터미널 로깅 데이터 출력
    print(f"입력된 토양/기후 데이터(N, Temp, pH, Rain): {custom_data[0]}")
    print(f"이 환경에 가장 어울리는 최적의 추천 작물은? : [{predicted_crop.upper()}] 입니다.\n")
    print(match_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

    # 브라우저 결과 JSON 출력 반환
    return {
        "status": "success",
        "input_environment": {
            "latitude": lat,
            "longitude": lon,
            "nitrogen_actual": nitrogen,
            "temperature_avg": temp,
            "ph": phh2o,
            "rainfall_scaled": precip
        },
        "best_recommended_crop": predicted_crop.upper(),
        "suitability_details": match_df.to_dict(orient="records")
    }

