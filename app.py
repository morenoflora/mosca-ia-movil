import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Monitor Bio-IA V14", page_icon="🪰", layout="wide")

st.title("🪰 Predictor de Dispersión y Carga Térmica (V14)")
st.markdown("Monitoreo de **19 GDD** y pronóstico de estrés térmico estival.")

# --- CARREGA MODEL ---
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load('best_model_lgbm_7d.pkl')
modelo = load_model()

# --- ESTADO DE SESIÓN ---
if 'lat' not in st.session_state: st.session_state.lat = 37.5
if 'lon' not in st.session_state: st.session_state.lon = -4.5

# --- MAPA ---
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=8)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Google', name='Google Hybrid', overlay=False, control=True
).add_to(m)
folium.Marker(location=[st.session_state.lat, st.session_state.lon], icon=folium.Icon(color='red')).add_to(m)

col_map, col_empty = st.columns([1.5, 1])
with col_map:
    map_out = st_folium(m, width=950, height=450, key="mapa_v14")

if map_out and map_out.get("last_clicked"):
    nl, ng = map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"]
    if nl != st.session_state.lat:
        st.session_state.lat, st.session_state.lon = nl, ng
        st.rerun()

st.sidebar.header("🕹️ Centro de Control")
st.sidebar.info(f"📍 Coords: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

# --- DESCARGA ---
def get_detailed_weather(lat, lon):
    hoy = datetime.date.today()
    inicio = datetime.date(hoy.year, 1, 1)
    
    # 1. Pronstico (Cubre 92 das atrs y 7 adelante)
    url_f = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=92&forecast_days=7&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    
    # 2. Histrico (Por si necesitamos ver algo de Enero previo a 90 das)
    url_a = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={inicio}&end_date={hoy - datetime.timedelta(days=10)}&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    
    try:
        r_f = requests.get(url_f).json()['daily']
        df = pd.DataFrame(r_f)
        
        # Opcionalmente parchar con archivo si falta Enero
        try:
            r_a = requests.get(url_a).json()['daily']
            df_a = pd.DataFrame(r_a)
            df = pd.concat([df_a, df]).drop_duplicates(subset=['time']).sort_values('time')
        except: pass
            
        df['date'] = pd.to_datetime(df['time']).dt.date
        df['deg_above_29'] = np.maximum(0, df['temperature_2m_max'] - 29)
        df['ADD29'] = df['deg_above_29'].cumsum()
        return df
    except: return None

if st.sidebar.button("🚀 Lanzar Predicción Térmica", type="primary"):
    with st.spinner("Analizando microclima y pronstico térmico..."):
        df = get_detailed_weather(st.session_state.lat, st.session_state.lon)
        if df is not None:
            hoy = datetime.date.today()
            # Datos de hoy
            actual = df[df['date'] <= hoy].iloc[-1]
            add29 = actual['ADD29']
            tm7 = df[df['date'] <= hoy]['temperature_2m_mean'].tail(7).mean()
            
            # Previsin
            fut = df[df['date'] > hoy]
            fc_t = fut['temperature_2m_mean'].mean()
            fc_add = fut['deg_above_29'].sum()
            fc_max = fut['temperature_2m_max'].max()
            
            # IA
            d19, is_d = 19 - add29, 1 if add29 >= 19 else 0
            X = pd.DataFrame([{
                'latitud': st.session_state.lat, 'longitud': st.session_state.lon, 'altura': 450.0,
                'ADD29': add29, 'dist_to_19_gdd': d19, 'is_dispersed': is_d,
                'tmean_roll_7': tm7, 'fcst_7d_tmean': fc_t, 'fcst_7d_ADD29_inc': fc_add
            }])
            prob = modelo.predict_proba(X)[0][1]
            
            # --- INTERFAZ ---
            st.divider()
            st.markdown(f"## 🔭 Informe Térmico: Proyección a 7 Días")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Acumulado GDD Actual", f"{add29:.1f} / 19", help="Grados da acumulados sobre 29C desde el 1 de enero.")
            col2.metric("🔥 Incremento de Calor (7d)", f"+{fc_add:.1f} GDD", delta_color="inverse", help="Grados da adicionales que se sumarn esta semana segn el satlite.")
            col3.metric("Probabilidad de Presencia", f"{prob:.1%}")
            
            if fc_add > 0:
                st.warning(f"⚠️ **Riesgo de Estrés Térmico**: Se prevé un aumento de **{fc_add:.1f} Grados-Día** esta semana. Máximas de hasta **{fc_max:.1f} ºC**.")
            else:
                st.info("ℹ️ Sin aumento de carga térmica previsto para la próxima semana (No se superan los 29ºC).")
            
            st.progress(min(1.0, add29/19))
            if add29 >= 19: st.error("🚨 **DISPERSIÓN ACTIVA**: La mosca ha superado el límite de confort biológico.")
            
            st.divider()
            st.subheader("📈 Evolución de los 19 GDD del Olivar")
            st.line_chart(df[df['date'] <= hoy].set_index('date')['ADD29'], color="#FF4B4B")
            st.caption("Grfica de acumulacin de estrés térmico (ADD29) en el ao actual.")
            
        else:
            st.error("Error al obtener los datos de la previsin.")
