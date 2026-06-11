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
import openai
import dotenv
import os

from starlette.responses import HTMLResponse

# API 호출 속도를 제어하기 위한 세마포어 (서버 차단 방지)
sem = asyncio.Semaphore(1)

# API 클라이언트 초기화 (API 키 입력)
client = openai.OpenAI(
    api_key=os.getenv("openai_api")
)

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


# ==========================================
# 5. HTML 대시보드 메인 라우터 (제공된 HTML 통합 및 Map 설정 보강)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def main():
    html_content = """
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>21세기 농사직설 — 위치 기반 작물 추천</title>

    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/sunn-us/SUIT-Variable/fonts/variable/woff2/SUIT-Variable.css">

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>

    <style>
    :root{
      --bg:#ffffff; --bg-soft:#f7f8f9; --bg-card:#ffffff;
      --ink:#191f28; --ink-2:#4e5968; --ink-3:#8b95a1;
      --line:#e5e8eb; --line-2:#f2f4f6;
      --green:#1b8a4c; --green-dark:#14703d; --green-bg:#eaf6ef;
      --blue:#3182f6;
      --r:12px; --r-sm:8px;
      --shadow:0 1px 3px rgba(0,0,0,.04), 0 6px 20px rgba(0,0,0,.06);
      --shadow-sm:0 1px 2px rgba(0,0,0,.04);
    }
    *{box-sizing:border-box; margin:0; padding:0}
    html,body{height:100%}
    body{font-family:"SUIT Variable","SUIT",-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
      color:var(--ink); background:var(--bg); line-height:1.6; -webkit-font-smoothing:antialiased}
    ::selection{background:#d3e9dd}
    button{font-family:inherit}

    .screen{display:none; min-height:100vh; flex-direction:column}
    .screen.active{display:flex; animation:fade .3s ease both}
    @keyframes fade{from{opacity:0} to{opacity:1}}

    /* ---------- 헤더 ---------- */
    .header{position:sticky; top:0; z-index:1000; background:rgba(255,255,255,.85);
      backdrop-filter:blur(12px); border-bottom:1px solid var(--line-2)}
    .header-in{max-width:1140px; margin:0 auto; padding:0 24px; height:64px;
      display:flex; align-items:center; justify-content:space-between}
    .logo{display:flex; align-items:center; gap:8px; font-weight:800; font-size:18px;
      letter-spacing:-.01em; cursor:pointer; color:var(--ink)}
    .logo svg{width:26px; height:26px}
    .nav-step{display:flex; align-items:center; gap:6px; font-size:13px; color:var(--ink-3); font-weight:500}
    .nav-step .now{color:var(--green); font-weight:700}
    .nav-step .sep{color:var(--line)}

    /* ---------- 버튼 ---------- */
    .btn{display:inline-flex; align-items:center; justify-content:center; gap:6px;
      border:none; cursor:pointer; font-size:15px; font-weight:600; padding:13px 22px;
      border-radius:var(--r-sm); background:var(--green); color:#fff; transition:.15s; letter-spacing:-.01em}
    .btn:hover{background:var(--green-dark)}
    .btn:active{transform:scale(.98)}
    .btn[disabled]{background:#d1d6db; cursor:not-allowed}
    .btn-lg{font-size:16px; padding:15px 28px}
    .btn-ghost{background:var(--bg-soft); color:var(--ink-2)}
    .btn-ghost:hover{background:var(--line-2); color:var(--ink)}

    /* ---------- 화면1 메인 ---------- */
    .hero{max-width:1140px; margin:0 auto; padding:88px 24px 72px; text-align:center}
    .hero .badge{display:inline-block; font-size:13px; font-weight:600; color:var(--green);
      background:var(--green-bg); padding:7px 14px; border-radius:100px; margin-bottom:22px}
    .hero h1{font-size:clamp(32px,4.6vw,52px); font-weight:800; letter-spacing:-.03em;
      line-height:1.25; margin-bottom:18px}
    .hero h1 .pt{color:var(--green)}
    .hero p{font-size:17px; color:var(--ink-2); max-width:34em; margin:0 auto 36px}
    .hero .cta-row{display:flex; gap:10px; justify-content:center; flex-wrap:wrap}

    .feature-strip{background:var(--bg-soft); border-top:1px solid var(--line-2); border-bottom:1px solid var(--line-2)}
    .features{max-width:1140px; margin:0 auto; padding:64px 24px;
      display:grid; grid-template-columns:repeat(3,1fr); gap:20px}
    .feature{background:var(--bg-card); border:1px solid var(--line); border-radius:var(--r);
      padding:28px 26px; box-shadow:var(--shadow-sm)}
    .feature .ic{width:44px; height:44px; border-radius:10px; background:var(--green-bg);
      display:grid; place-items:center; font-size:21px; margin-bottom:16px}
    .feature h3{font-size:17px; font-weight:700; letter-spacing:-.01em; margin-bottom:6px}
    .feature p{font-size:14px; color:var(--ink-2)}

    .hero-foot{max-width:1140px; margin:0 auto; padding:48px 24px 80px; text-align:center}
    .hero-foot p{font-size:14px; color:var(--ink-3)}

    /* ---------- 화면2 지도 ---------- */
    .map-body{flex:1; max-width:1140px; margin:0 auto; width:100%; padding:32px 24px 48px;
      display:grid; grid-template-columns:340px 1fr; gap:20px}
    .panel{align-self:start; position:sticky; top:88px}
    .panel-card{background:var(--bg-card); border:1px solid var(--line); border-radius:var(--r);
      padding:24px; box-shadow:var(--shadow-sm)}
    .panel h2{font-size:20px; font-weight:700; letter-spacing:-.02em; margin-bottom:6px}
    .panel .guide{font-size:14px; color:var(--ink-2); margin-bottom:20px}
    .sel-box{border:1px solid var(--line); border-radius:var(--r-sm); background:var(--bg-soft); padding:18px}
    .sel-box .sl{font-size:12px; font-weight:600; color:var(--ink-3); margin-bottom:6px}
    .sel-box .sn{font-size:19px; font-weight:700; letter-spacing:-.01em; word-break:keep-all}
    .sel-box .sg{font-size:13px; color:var(--ink-3); margin-top:2px; font-variant-numeric:tabular-nums}
    .sel-box .btn{width:100%; margin-top:16px}

    .map-stage{position:relative; border-radius:var(--r); overflow:hidden; min-height:560px;
      border:1px solid var(--line); box-shadow:var(--shadow-sm); background:var(--bg-soft)}
    #gmap{position:absolute; inset:0; z-index:1}
    .manual-btn{position:absolute; right:14px; top:14px; z-index:1005; border:1px solid var(--line);
      background:#fff; color:var(--ink-2); font-weight:600; font-size:13px; padding:9px 14px;
      border-radius:var(--r-sm); cursor:pointer; display:flex; align-items:center; gap:6px;
      box-shadow:var(--shadow-sm); transition:.15s}
    .manual-btn:hover{border-color:var(--ink-3); color:var(--ink)}
    .map-hint{position:absolute; left:14px; bottom:14px; z-index:1004; font-size:13px; color:var(--ink-2);
      background:rgba(255,255,255,.92); padding:8px 14px; border-radius:var(--r-sm);
      border:1px solid var(--line); box-shadow:var(--shadow-sm)}
    .map-loading{position:absolute; inset:0; display:none; place-items:center; z-index:1006;
      color:#fff; font-size:16px; font-weight:600; background:rgba(0,0,0,0.5)}

    /* ---------- 화면3 결과 ---------- */
    .result-body{flex:1; max-width:1140px; margin:0 auto; width:100%; padding:32px 24px 72px}
    .back-link{display:inline-flex; align-items:center; gap:6px; font-size:14px; font-weight:600;
      color:var(--ink-2); cursor:pointer; background:none; border:none; padding:8px 0; margin-bottom:8px; transition:.15s}
    .back-link:hover{color:var(--ink)}
    .result-head .done{display:inline-flex; align-items:center; gap:6px; font-size:13px; font-weight:600;
      color:var(--green); background:var(--green-bg); padding:6px 12px; border-radius:100px; margin-bottom:12px}
    .result-head h1{font-size:clamp(24px,3.4vw,34px); font-weight:800; letter-spacing:-.02em;
      line-height:1.3; margin-bottom:28px}
    .result-head h1 .pt{color:var(--green)}

    .chips{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:36px}
    .chip{background:var(--bg-card); border:1px solid var(--line); border-radius:var(--r);
      padding:18px 20px; box-shadow:var(--shadow-sm)}
    .chip .cl{font-size:13px; color:var(--ink-3); font-weight:500}
    .chip .cv{font-size:26px; font-weight:800; letter-spacing:-.02em; margin-top:4px;
      font-variant-numeric:tabular-nums}
    .chip .cv small{font-size:13px; font-weight:500; color:var(--ink-3); margin-left:2px}

    .sect-title{font-size:18px; font-weight:700; letter-spacing:-.01em; margin-bottom:14px}

    .crop-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px}
    .crop-card{position:relative; background:var(--bg-card); border:1px solid var(--line);
      border-radius:var(--r); padding:22px; box-shadow:var(--shadow-sm); transition:.15s; cursor:pointer}
    .crop-card:hover{border-color:#c6cbd1; box-shadow:var(--shadow)}
    .crop-card.top{border:1.5px solid var(--green)}
    .crop-card .best{position:absolute; right:16px; top:16px; font-size:11px; font-weight:700;
      color:#fff; background:var(--green); padding:4px 9px; border-radius:6px}
    .crop-card .rk{font-size:13px; font-weight:600; color:var(--ink-3); margin-bottom:10px}
    .crop-card .head-row{display:flex; align-items:center; gap:12px; margin-bottom:4px}
    .crop-card .emoji{font-size:32px; line-height:1}
    .crop-card h3{font-size:18px; font-weight:700; letter-spacing:-.01em}
    .crop-card .sci{font-size:13px; color:var(--ink-3); margin-bottom:16px}
    .bar-row{display:flex; align-items:center; justify-content:space-between; margin-bottom:6px}
    .bar-row .bl{font-size:12px; color:var(--ink-3); font-weight:500}
    .bar-row .bv{font-size:14px; font-weight:700; color:var(--green); font-variant-numeric:tabular-nums}
    .bar{height:6px; border-radius:100px; background:var(--line-2); overflow:hidden}
    .bar i{display:block; height:100%; border-radius:100px; width:0; background:var(--green);
      transition:width .9s cubic-bezier(.2,.8,.2,1)}
    .tags{display:flex; flex-wrap:wrap; gap:6px; margin-top:14px}
    .tag{font-size:12px; font-weight:600; color:var(--ink-2); background:var(--bg-soft);
      padding:5px 10px; border-radius:6px; border:1px solid var(--line)}
    .tag.ok{color:var(--green); background:var(--green-bg); border-color:#cfe8da}
    .crop-card .more{margin-top:16px; font-size:13px; font-weight:600; color:var(--blue);
      display:flex; align-items:center; gap:4px}

    .footer{border-top:1px solid var(--line-2); margin-top:auto}
    .footer-in{max-width:1140px; margin:0 auto; padding:28px 24px; font-size:13px; color:var(--ink-3);
      display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px}

    /* ---------- 모달 공통 ---------- */
    .modal-overlay{position:fixed; inset:0; z-index:2000; display:none; place-items:center;
      background:rgba(25,31,40,.5); padding:20px}
    .modal-overlay.open{display:grid; animation:fade .2s ease both}
    .modal{background:#fff; border-radius:16px; max-width:560px; width:100%; max-height:88vh;
      overflow:auto; position:relative; padding:28px; box-shadow:0 20px 60px rgba(0,0,0,.2);
      animation:pop .25s cubic-bezier(.2,.8,.2,1) both}
    @keyframes pop{from{opacity:0; transform:translateY(12px) scale(.98)} to{opacity:1; transform:none}}
    .m-close{position:absolute; right:14px; top:14px; width:34px; height:34px; border:none;
      background:var(--bg-soft); border-radius:8px; cursor:pointer; font-size:15px; color:var(--ink-2);
      display:grid; place-items:center; transition:.15s}
    .m-close:hover{background:var(--line-2); color:var(--ink)}

    /* 작물 모달 */
    .modal .m-top{display:flex; align-items:center; gap:14px; margin-bottom:4px}
    .modal .m-emoji{font-size:40px; line-height:1}
    .modal h3{font-size:22px; font-weight:800; letter-spacing:-.01em}
    .modal .m-sci{font-size:13px; color:var(--ink-3)}
    .modal .m-meta{display:flex; gap:8px; flex-wrap:wrap; margin:14px 0 4px}
    .m-pill{font-size:12px; font-weight:600; padding:5px 11px; border-radius:6px;
      background:var(--bg-soft); color:var(--ink-2); border:1px solid var(--line)}
    .m-pill.g{background:var(--green-bg); color:var(--green); border-color:#cfe8da}
    .m-body{margin-top:14px; border-top:1px solid var(--line-2); padding-top:4px}
    .m-body h4{font-size:15px; font-weight:700; color:var(--ink); margin:16px 0 6px}
    .m-body p{font-size:14px; color:var(--ink-2); margin:4px 0}
    .m-body ul{list-style:none; margin:4px 0; padding:0}
    .m-body li{font-size:14px; color:var(--ink-2); position:relative; padding-left:14px; margin:5px 0}
    .m-body li::before{content:""; position:absolute; left:2px; top:9px; width:5px; height:5px;
      border-radius:50%; background:var(--green)}
    .m-body strong{color:var(--ink); font-weight:700}
    .ai-loading{display:flex; align-items:center; gap:10px; font-size:14px; color:var(--ink-2); padding:20px 0}
    .ai-loading .spin{width:16px; height:16px; border:2px solid var(--line);
      border-top-color:var(--green); border-radius:50%; animation:spin .8s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .m-note{margin-top:16px; font-size:12px; color:var(--ink-3); line-height:1.5;
      border-top:1px solid var(--line-2); padding-top:12px}

    /* 수동 입력 모달 */
    .input-box{max-width:420px}
    .input-box h3{font-size:19px; font-weight:700; margin-bottom:4px}
    .input-box>p{font-size:14px; color:var(--ink-2); margin-bottom:18px}
    .in-label{display:block; font-size:13px; font-weight:600; color:var(--ink-2); margin:0 0 6px 1px}
    .in-row{display:grid; grid-template-columns:1fr 1fr; gap:10px}
    .input-field{width:100%; font-size:15px; color:var(--ink); padding:12px 14px;
      border:1px solid var(--line); border-radius:var(--r-sm); background:#fff; transition:.15s}
    .input-field:focus{outline:none; border-color:var(--green); box-shadow:0 0 0 3px rgba(27,138,76,.12)}
    .input-msg{min-height:18px; font-size:13px; color:#e5484d; margin:8px 1px 0}
    .input-box .btn{width:100%; margin-top:12px}

    @media(max-width:900px){
      .features{grid-template-columns:1fr}
      .map-body{grid-template-columns:1fr}
      .panel{position:static}
      .map-stage{min-height:420px}
      .chips{grid-template-columns:repeat(2,1fr)}
      .crop-grid{grid-template-columns:1fr}
      .hero{padding:56px 24px 48px}
    }
    @media(max-width:520px){
      .chips{grid-template-columns:repeat(2,1fr)}
      .btn-lg{width:100%}
    }
    </style>

    <section class="screen active" id="main">
      <header class="header">
        <div class="header-in">
          <div class="logo" onclick="go('main')">
            <svg viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="6" fill="#1b8a4c"/><path d="M12 17v-5m0 0c0-2.5 1.8-4.5 4.5-4.5 0 2.5-1.8 4.5-4.5 4.5Zm0 0c0-2-1.5-3.6-3.6-3.6 0 2 1.5 3.6 3.6 3.6Z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            21세기 농사직설
          </div>
          <div class="nav-step"><span class="now">1 시작</span><span class="sep">·</span><span>2 위치 선택</span><span class="sep">·</span><span>3 추천 결과</span></div>
        </div>
      </header>

      <div class="hero">
        <span class="badge">위치 기반 작물 추천 서비스</span>
        <h1>땅을 고르면<br>가장 <span class="pt">잘 자랄 작물</span>을 알려드려요</h1>
        <p>지도를 클릭하면 그 지점의 토양·기온·강수 환경을 분석해 어울리는 작물을 추천하고,
          작물별 재배법과 지역 맞춤 조언까지 제공합니다.</p>
        <div class="cta-row">
          <button class="btn btn-lg" onclick="go('map')">지도에서 위치 고르기</button>
        </div>
      </div>

      <div class="feature-strip">
        <div class="features">
          <div class="feature">
            <div class="ic">📍</div>
            <h3>위치 선택</h3>
            <p>전 세계 지도를 클릭하거나 위도·경도를 직접 입력해 분석 지점을 정합니다.</p>
          </div>
          <div class="feature">
            <div class="ic">📊</div>
            <h3>환경 분석</h3>
            <p>좌표 기반으로 토양 pH, 평균 기온, 연 강수량, 질소 함량을 추정합니다.</p>
          </div>
          <div class="feature">
            <div class="ic">🌱</div>
            <h3>작물 추천 · 재배 조언</h3>
            <p>적합도 순으로 작물을 추천하고, AI가 지역 맞춤 재배법을 알려드립니다.</p>
          </div>
        </div>
      </div>

      <div class="hero-foot">
        <p>회원가입 없이 바로 사용할 수 있습니다.</p>
      </div>

      <footer class="footer">
        <div class="footer-in"><span>© 21세기 농사직설</span><span>좌표 기반 환경 추정 데모</span></div>
      </footer>
    </section>

    <section class="screen" id="map">
      <header class="header">
        <div class="header-in">
          <div class="logo" onclick="go('main')">
            <svg viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="6" fill="#1b8a4c"/><path d="M12 17v-5m0 0c0-2.5 1.8-4.5 4.5-4.5 0 2.5-1.8 4.5-4.5 4.5Zm0 0c0-2-1.5-3.6-3.6-3.6 0 2 1.5 3.6 3.6 3.6Z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            21세기 농사직설
          </div>
          <div class="nav-step"><span>1 시작</span><span class="sep">·</span><span class="now">2 위치 선택</span><span class="sep">·</span><span>3 추천 결과</span></div>
        </div>
      </header>

      <div class="map-body">
        <aside class="panel">
          <div class="panel-card">
            <h2>어디를 살펴볼까요?</h2>
            <p class="guide">지도를 클릭하면 해당 지점의 좌표를 가져와 환경 분석을 준비합니다.</p>
            <div class="sel-box">
              <div class="sl">선택된 위치</div>
              <div class="sn" id="sbName">아직 선택되지 않음</div>
              <div class="sg" id="sbGeo">지도를 클릭해 주세요</div>
              <button class="btn" id="goBtn" onclick="requestPrediction()" disabled>추천 받기</button>
            </div>
          </div>
        </aside>
        <div class="map-stage">
          <div id="gmap"></div>
          <div class="map-loading" id="mapLoading">데이터를 수집하고 추론하는 중입니다. 대략 5초 가량 소요됩니다...</div>
          <button class="manual-btn" onclick="openInput()">좌표 직접 입력</button>
          <div class="map-hint">지도를 클릭해 위치를 선택하세요</div>
        </div>
      </div>

      <footer class="footer">
        <div class="footer-in"><span>© 21세기 농사직설</span><span>좌표 기반 환경 추정 데모</span></div>
      </footer>
    </section>

    <section class="screen" id="result">
      <header class="header">
        <div class="header-in">
          <div class="logo" onclick="go('main')">
            <svg viewBox="0 0 24 24" fill="none"><rect width="24" height="24" rx="6" fill="#1b8a4c"/><path d="M12 17v-5m0 0c0-2.5 1.8-4.5 4.5-4.5 0 2.5-1.8 4.5-4.5 4.5Zm0 0c0-2-1.5-3.6-3.6-3.6 0 2 1.5 3.6 3.6 3.6Z" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            21세기 농사직설
          </div>
          <div class="nav-step"><span>1 시작</span><span class="sep">·</span><span>2 위치 선택</span><span class="sep">·</span><span class="now">3 추천 결과</span></div>
        </div>
      </header>

      <div class="result-body">
        <button class="back-link" onclick="go('map')">← 다른 위치 선택</button>
        <div class="result-head">
          <div><span class="done">✓ 분석 완료</span></div>
          <h1><span class="pt" id="resRegion">선택 지역</span>에 잘 맞는 작물</h1>
        </div>

        <div class="chips" id="chips"></div>

        <div class="sect-title">추천 작물</div>
        <div class="crop-grid" id="cropGrid"></div>
      </div>

      <footer class="footer">
        <div class="footer-in"><span>© 21세기 농사직설</span><span>좌표 기반 환경 추정 데모 · 작물을 누르면 상세 재배 조언을 볼 수 있어요</span></div>
      </footer>
    </section>

    <div class="modal-overlay" id="modal" onclick="if(event.target===this) closeCrop()">
      <div class="modal" id="modalCard"></div>
    </div>

    <div class="modal-overlay" id="inputModal" onclick="if(event.target===this) closeInput()">
      <div class="modal input-box">
        <button class="m-close" onclick="closeInput()" aria-label="닫기">✕</button>
        <h3>좌표 직접 입력</h3>
        <p>위도와 경도를 입력하세요.</p>
        <div class="in-row">
          <div>
            <label class="in-label" for="inLat">위도 (Latitude)</label>
            <input id="inLat" class="input-field" inputmode="decimal" placeholder="35.18">
          </div>
          <div>
            <label class="in-label" for="inLng">경도 (Longitude)</label>
            <input id="inLng" class="input-field" inputmode="decimal" placeholder="129.07">
          </div>
        </div>
        <div class="input-msg" id="inputMsg"></div>
        <button class="btn" onclick="submitInput()">이 위치로 선택</button>
      </div>
    </div>

    <script>
    /* ============================================================
       결과 칩 라벨 (env 배열 순서와 1:1 대응: [pH, 기온, 강수량, 질소])
       ============================================================ */
    const CHIP_META = [
      {label:"토양 pH", unit:""},
      {label:"평균 기온", unit:"℃"},
      {label:"연 강수량", unit:"mm"},
      {label:"토양 질소", unit:"mg/kg"},
    ];

    /* ============================================================
       작물 풀 (22종 확장 테이블 예시 맵핑 구조)
       ============================================================ */
    const CROP_POOL = {
      "rice": {n:"벼", e:"🌾", s:"Oryza sativa"},
      "maize": {n:"옥수수", e:"🌽", s:"Zea mays"},
      "chickpea": {n:"병아리콩", e:"🫛", s:"Cicer arietinum"},
      "kidneybeans": {n:"강낭콩", e:"🫘", s:"Phaseolus vulgaris"},
      "pigeonpeas": {n:"비둘기콩", e:"🫛", s:"Cajanus cajan"},
      "mothbeans": {n:"모스빈", e:"🫘", s:"Vigna aconitifolia"},
      "mungbean": {n:"녹두", e:"🟢", s:"Vigna radiata"},
      "blackgram": {n:"검은녹두", e:"⚫", s:"Vigna mungo"},
      "lentil": {n:"렌틸콩", e:"🟤", s:"Lens culinaris"},
      "pomegranate": {n:"석류", e:"🔴", s:"Punica granatum"},
      "banana": {n:"바나나", e:"🍌", s:"Musa acuminata"},
      "mango": {n:"망고", e:"🥭", s:"Mangifera indica"},
      "grapes": {n:"포도", e:"🍇", s:"Vitis vinifera"},
      "watermelon": {n:"수박", e:"🍉", s:"Citrullus lanatus"},
      "muskmelon": {n:"멜론", e:"🍈", s:"Cucumis melo"},
      "apple": {n:"사과", e:"🍎", s:"Malus domestica"},
      "orange": {n:"오렌지", e:"🍊", s:"Citrus sinensis"},
      "papaya": {n:"파파야", e:"🧆", s:"Carica papaya"},
      "coconut": {n:"코코넛", e:"🥥", s:"Cocos nucifera"},
      "cotton": {n:"면화", e:"Field", s:"Gossypium hirsutum"},
      "jute": {n:"황마", e:"🌱", s:"Corchorus olitorius"},
      "coffee": {n:"커피", e:"☕", s:"Coffea arabica"}
    };

    let map;
    let selectedLat = null;
    let selectedLon = null;
    let marker = null;

    // 화면 전환 제어 함수
    function go(screenId) {
      document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
      document.getElementById(screenId).classList.add('active');

      if (screenId === 'map') {
        // Leaflet Map 레이아웃 리프레시 목적 초기화 격리 보장
        setTimeout(() => {
          if (!map) {
            map = L.map('gmap').setView([36.5, 127.5], 7); // 대한민국 중심 기본 세팅
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '© OpenStreetMap contributors'
            }).addTo(map);

            map.on('click', function(e) {
              updateSelection(e.latlng.lat, e.latlng.lng);
            });
          } else {
            map.invalidateSize();
          }
        }, 200);
      }
    }

    // 마커 분기 업데이트 및 상태 바인딩 제어
    function updateSelection(lat, lon) {
      selectedLat = parseFloat(lat).toFixed(4);
      selectedLon = parseFloat(lon).toFixed(4);

      document.getElementById('sbName').innerText = `지정 좌표 확인`;
      document.getElementById('sbGeo').innerText = `위도: ${selectedLat}, 경도: ${selectedLon}`;
      document.getElementById('goBtn').removeAttribute('disabled');

      if (marker) {
        marker.setLatLng([lat, lon]);
      } else {
        marker = L.marker([lat, lon]).addTo(map);
      }
    }

    // 수동 좌표 입력 제어
    function openInput() { document.getElementById('inputModal').classList.add('open'); }
    function closeInput() { document.getElementById('inputModal').classList.remove('open'); }

    function submitInput() {
      const latVal = document.getElementById('inLat').value;
      const lngVal = document.getElementById('inLng').value;
      if(!latVal || !lngVal) {
        document.getElementById('inputMsg').innerText = "위도와 경도를 모두 채워주세요.";
        return;
      }
      updateSelection(latVal, lngVal);
      map.setView([latVal, lngVal], 10);
      closeInput();
    }

    // 비동기 AI 머신러닝 추론 데이터 요청 
    async function requestPrediction() {
      if(!selectedLat || !selectedLon) return;

      const loadingMask = document.getElementById('mapLoading');
      loadingMask.style.display = 'grid';

      try {
        const response = await fetch(`/predict/${selectedLat}/${selectedLon}`);
        const data = await response.json();

        if (data.status === 'success') {
          renderResults(data);
          go('result');
        } else {
          alert("에러가 발생했습니다: " + data.message);
        }
      } catch (err) {
        console.error(err);
        alert("서버 통신 중 장애가 발생했습니다.");
      } finally {
        loadingMask.style.display = 'none';
      }
    }

    // 최종 결과 템플릿 마크업 바인딩
    function renderResults(data) {
      document.getElementById('resRegion').innerText = `위도 ${data.input_environment.latitude}, 경도 ${data.input_environment.longitude}`;

      // 상단 지표 세팅
      const env = data.input_environment;
      const chipsContainer = document.getElementById('chips');
      chipsContainer.innerHTML = `
        <div class="chip"><div class="cl">${CHIP_META[0].label}</div><div class="cv">${env.ph}</div></div>
        <div class="chip"><div class="cl">${CHIP_META[1].label}</div><div class="cv">${env.temperature_avg}<small>${CHIP_META[1].unit}</small></div></div>
        <div class="chip"><div class="cl">${CHIP_META[2].label}</div><div class="cv">${env.rainfall_scaled}<small>${CHIP_META[2].unit}</small></div></div>
        <div class="chip"><div class="cl">${CHIP_META[3].label}</div><div class="cv">${env.nitrogen_actual.toFixed(1)}<small>${CHIP_META[3].unit}</small></div></div>
      `;

      // 추천 카드 구성 리스트업
      const grid = document.getElementById('cropGrid');
      grid.innerHTML = "";

      data.suitability_details.forEach((item, index) => {
        const rawKey = item['작물 (Crop)'].toLowerCase().replace(/\s+/g, '');
        const meta = CROP_POOL[rawKey] || {n: item['작물 (Crop)'], e: "🌱", s: "Unknown genus"};
        const matchPct = item['추천 적합도 (Match %)'];

        // 상위 12개 가량 유의미한 항목 정렬 노출 필터링
        if(matchPct < 0.1 && index > 3) return;

        const isTop = index === 0;
        const card = document.createElement('div');
        card.className = `crop-card ${isTop ? 'top' : ''}`;
        card.onclick = () => openCropDetails(meta, matchPct, env);

        card.innerHTML = `
          ${isTop ? '<span class="best">최적 추천</span>' : ''}
          <div class="rk">순위 ${index + 1}</div>
          <div class="head-row">
            <span class="emoji">${meta.e}</span>
            <div>
              <h3>${meta.n}</h3>
              <div class="sci">${meta.s}</div>
            </div>
          </div>
          <div class="bar-row">
            <span class="bl">알고리즘 적합도</span>
            <span class="bv">${matchPct}%</span>
          </div>
          <div class="bar"><i style="width: ${matchPct}%"></i></div>
          <div class="more">상세 정보 및 재배 팁 보기 →</div>
        `;
        grid.appendChild(card);
      });
    }

    // 모달창 오픈 및 제어 스크립트
    function openCropDetails(meta, matchPct, env) {
      const modal = document.getElementById('modal');
      const modalCard = document.getElementById('modalCard');
      modal.classList.add('open');

      modalCard.innerHTML = `
        <button class="m-close" onclick="closeCrop()">✕</button>
        <div class="m-top">
          <span class="m-emoji">${meta.e}</span>
          <div>
            <h3>${meta.n}</h3>
            <div class="m-sci">${meta.s}</div>
          </div>
        </div>
        <div class="m-meta">
          <span class="m-pill g">추천 적합도 ${matchPct}%</span>
        </div>
        <div class="m-body">
          <h4>🌱 기본 재배 환경 정보</h4>
          <p>작물의 고유 성질에 부합하는 적합성 예측 분석을 완료했습니다.</p>
          <ul>
            <li>현재 분석지의 pH 상태(<strong>${env.ph}</strong>) 및 평균 기온(<strong>${env.temperature_avg}℃</strong>) 정보와 결합하여 농업 생산성을 극대화할 수 있는 관리 스케줄 작성을 권장합니다.</li>
          </ul>
          <div class="m-note">본 시스템의 예측값은 다변량 기후 데이터 원격 탐사 모델링 추정치이므로, 실제 필지 환경과 일부 차이가 발생할 수 있습니다.</div>
        </div>
      `;
    }

    function closeCrop() { document.getElementById('modal').classList.remove('open'); }
    </script>
    """
    return HTMLResponse(content=html_content)