import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Monitor Bio-IA V12", page_icon="🪰", layout="wide")

st.title("🪰 Monitor de Dispersión y Previsión a 7 Días (V12)")
st.markdown("""
Analiza la **dispersión estival** y la **probabilidad de vuelo** en tiempo real. 
*Actualizado para soportar acumulados de todo el año.*
""")

# --- CARREGA MODEL ---
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load('best_model_lgbm_7d.pkl')
modelo = load_model()

# --- ESTADO DE SESIÓN ---
if 'lat' not in st.session_state: st.session_state.lat = 37.5
if 'lon' not in st.session_state: st.session_state.lon = -4.5
if 'zoom' not in st.session_state: st.session_state.zoom = 8

# --- MAPA SATELITAL ---
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=st.session_state.zoom)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Google', name='Google Hybrid (Detalle)', overlay=False, control=True
).add_to(m)
folium.Marker(location=[st.session_state.lat, st.session_state.lon], icon=folium.Icon(color='red')).add_to(m)

col_map, col_empty = st.columns([1.5, 1])
with col_map:
    output = st_folium(m, width=950, height=500, key="mapa_v12")

if output.get("last_clicked"):
    n_lat, n_lon = output["last_clicked"]["lat"], output["last_clicked"]["lng"]
    if n_lat != st.session_state.lat:
        st.session_state.lat, st.session_state.lon = n_lat, n_lon
        st.session_state.zoom = output.get("zoom", 8)
        st.rerun()

# ---SIDEBAR ---
st.sidebar.header("🕹️ Centro de Control")
st.sidebar.info(f"📍 Lat: {st.session_state.lat:.4f} | Lon: {st.session_state.lon:.4f}")

# --- FUNCIÓN DE DESCARGA HÍBRIDA ---
def get_weather_data(lat, lon):
    hoy = datetime.date.today()
    inicio_ano = datetime.date(hoy.year, 1, 1)
    hace_2_dias = hoy - datetime.timedelta(days=2)
    
    # 1. Descargar Histórico (Desde el 1 de Enero hasta hace 2 días)
    url_hist = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={inicio_ano}&end_date={hace_2_dias}&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    
    # 2. Descargar Forecast (Desde hace 1 día hasta +7 días)
    url_fore = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=1&forecast_days=7&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    
    try:
        r_hist = requests.get(url_hist).json()['daily']
        r_fore = requests.get(url_fore).json()['daily']
        
        df_h = pd.DataFrame(r_hist)
        df_f = pd.DataFrame(r_fore)
        
        # Combinar y quitar duplicados (por si se solapan)
        df = pd.concat([df_h, df_f]).drop_duplicates(subset=['time']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['time']).dt.date
        return df
    except Exception as e:
        st.error(f"Error de conexión con satélites: {e}")
        return None

if st.sidebar.button("🚀 Lanzar Predicción a 7 Días", type="primary", use_container_width=True):
    with st.spinner("Analizando ciclo biológico y térmico..."):
        df = get_weather_data(st.session_state.lat, st.session_state.lon)
        
        if df is not None:
            hoy = datetime.date.today()
            df['deg_above_29'] = np.maximum(0, df['temperature_2m_max'] - 29)
            df['ADD29'] = df['deg_above_29'].cumsum()
            
            # Extraer métricas clave
            current_row = df[df['date'] <= hoy].iloc[-1]
            add29 = current_row['ADD29']
            tm7 = df[df['date'] <= hoy]['temperature_2m_mean'].tail(7).mean()
            
            df_future = df[df['date'] > hoy]
            fcst_t = df_future['temperature_2m_mean'].mean()
            fcst_add = df_future['deg_above_29'].sum()
            
            # Predicción IA
            dist = 19 - add29
            is_disp = 1 if add29 >= 19 else 0
            X = pd.DataFrame([{
                'latitud': st.session_state.lat, 'longitud': st.session_state.lon, 'altura': 450.0,
                'ADD29': add29, 'dist_to_19_gdd': dist, 'is_dispersed': is_disp,
                'tmean_roll_7': tm7, 'fcst_7d_tmean': fcst_t, 'fcst_7d_ADD29_inc': fcst_add
            }])
            prob = modelo.predict_proba(X)[0][1]
            
            # --- INTERFAZ ---
            st.divider()
            st.markdown(f"## 🔭 Informe de Riesgo (Periodo: +7 días)")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 Actividad Próxima Semana")
                st.metric("Probabilidad de Vuelo", f"{prob:.1%}")
                if prob > 0.6: st.warning("⚠️ Probabilidad ALTA")
            with c2:
                st.subheader("🚶 Umbral Dispersión (19 GDD)")
                st.metric("Acumulado", f"{add29:.1f} / 19")
                st.progress(min(1.0, add29/19))
            
            st.divider()
            st.subheader("📈 Evolución térmica")
            st.line_chart(df[df['date'] <= hoy].set_index('date')['ADD29'])
            
        else:
            st.error("No se pudieron obtener datos para esta ubicación.")
