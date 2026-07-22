import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. KOCAELİ MİKRO-KLİMA LOKASYON VERİLERİ
# ==========================================
# 12 Temsilci nokta: Enlem, Boylam, Rakım (m) ve Denize Mesafe (km)
kocaeli_locations = {
    "İzmit": {"lat": 40.7654, "lon": 29.9408, "elevation": 10, "dist_to_sea": 0.5},
    "Kartepe Zirve": {"lat": 40.6700, "lon": 30.0200, "elevation": 1150, "dist_to_sea": 22.0},
    "Kandıra Sahil": {"lat": 41.0711, "lon": 30.1500, "elevation": 15, "dist_to_sea": 0.2},
    "Gebze": {"lat": 40.8028, "lon": 29.4307, "elevation": 140, "dist_to_sea": 3.5},
    "Gölcük": {"lat": 40.4323, "lon": 29.8251, "elevation": 20, "dist_to_sea": 0.3},
    "Karamürsel": {"lat": 40.6922, "lon": 29.6161, "elevation": 15, "dist_to_sea": 0.2},
    "Başiskele": {"lat": 40.7100, "lon": 29.9300, "elevation": 60, "dist_to_sea": 2.0},
    "Derince": {"lat": 40.7562, "lon": 29.8311, "elevation": 40, "dist_to_sea": 1.0},
    "Körfez": {"lat": 40.7744, "lon": 29.7369, "elevation": 30, "dist_to_sea": 0.8},
    "Dilovası": {"lat": 40.7872, "lon": 29.5469, "elevation": 80, "dist_to_sea": 2.5},
    "Çayırova": {"lat": 40.8164, "lon": 29.3736, "elevation": 110, "dist_to_sea": 5.0},
    "Yuvacık Baraj Çevresi": {"lat": 40.6500, "lon": 29.9500, "elevation": 220, "dist_to_sea": 14.0}
}

# ==========================================
# 2. OPEN-METEO API'DEN VERİ ÇEKME FONKSİYONU
# ==========================================
def fetch_location_data(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "hourly": "relative_humidity_2m,temperature_2m,wind_speed_10m,surface_pressure"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    return pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "humidity": data["hourly"]["relative_humidity_2m"],
        "temperature": data["hourly"]["temperature_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

# Tüm lokasyon verilerini toplama
df_list = []
print("Kocaeli genelinden iklim verileri toplanıyor...")
for name, info in kocaeli_locations.items():
    df = fetch_location_data(info["lat"], info["lon"])
    df["location"] = name
    df["elevation"] = info["elevation"]
    df["dist_to_sea"] = info["dist_to_sea"]
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour
    df_list.append(df)

full_dataset = pd.concat(df_list, ignore_index=True)
print(f"Toplam Veri Sayısı: {len(full_dataset)} satır.")

# ==========================================
# 3. VERİ HAZIRLAMA VE MODEL EĞİTİMİ
# ==========================================
# Girdi Özellikleri (Features) - Veri sızıntısını önlemek için dew_point dahil edilmedi.
X = full_dataset[["temperature", "wind_speed", "pressure", "elevation", "dist_to_sea", "month", "hour"]]
y = full_dataset["humidity"]

# Eğitim ve Test Seti Ayrımı (%80 Train, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Modeli
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==========================================
# 4. TAHMİN VE PERFORMANS DEĞERLENDİRMESİ
# ==========================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL BAŞARI METRİKLERİ ---")
print(f"Mean Absolute Error (MAE): %{mae:.2f} nem")
print(f"R² Score: %{r2 * 100:.2f}")

# ==========================================
# 5. ÖRNEK TAHMİN SENARYOSU
# ==========================================
sample_input = pd.DataFrame([{
    "temperature": 22.0,
    "wind_speed": 12.0,
    "pressure": 980.0,
    "elevation": 500,
    "dist_to_sea": 15.0,
    "month": 6,
    "hour": 14
}])

predicted_humidity = model.predict(sample_input)[0]
print(f"\nÖrnek Senaryo Tahmin Edilen Nem: %{predicted_humidity:.2f}")
