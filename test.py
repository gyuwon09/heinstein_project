import requests
import json
from datetime import datetime, timedelta  # 날짜 자동 계산을 위해 추가

def get_nasa_power_data(lat, lon):
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
    print(f"    위도: {latitude}, 경도: {longitude} 지역의 데이터를 NASA POWER API에서 불러오는 중...")

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


if __name__ == "__main__":
    latitude = 37.5665
    longitude = 126.9780

    data = get_nasa_power_data(latitude, longitude)
