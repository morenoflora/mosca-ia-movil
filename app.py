import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Monitor Bio-IA V11", page_icon="🪰", layout="wide")

st.title("🪰 Monitor de Dispersión y Previsión a 7 Días")
st.markdown("""
Analiza la **dispersión estival** y la **probabilidad de vuelo para la próxima semana** en cualquier punto del mapa.
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
    output = st_folium(m, width=950, height=500, key="mapa_v11")

if output.get("last_clicked"):
    n_lat, n_lon = output["last_clicked"]["lat"], output["last_clicked"]["lng"]
    if n_lat != st.session_state.lat:
        st.session_state.lat, st.session_state.lon = n_lat, n_lon
        st.session_state.zoom = output.get("zoom", 8)
        st.rerun()

# --- SIDEBAR & ACCIÓN ---
st.sidebar.header("🕹️ Centro de Control")
st.sidebar.info(f"📍 Lat: {st.session_state.lat:.4f} | Lon: {st.session_state.lon:.4f}")

if st.sidebar.button("🚀 Lanzar Predicción a 7 Días", type="primary", use_container_width=True):
    with st.spinner("Analizando ciclo biol\xf3gico..."):
        # API Online
        hoy = datetime.date.today()
        url = f"https://api.open-meteo.com/v1/forecast?latitude={st.session_state.lat}&longitude={st.session_state.lon}&start_date={datetime.date(hoy.year,1,1)}&end_date={hoy + datetime.timedelta(days=7)}&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
        try:
            r = requests.get(url).json()['daily']
            df = pd.DataFrame(r)
            df['date'] = pd.to_datetime(df['time'])
            df['deg_above_29'] = np.maximum(0, df['temperature_2m_max'] - 29)
            df['ADD29'] = df['deg_above_29'].cumsum()
            
            # Datos 2026
            add29 = df[df['date'].dt.date <= hoy]['ADD29'].iloc[-1]
            tm7 = df[df['date'].dt.date <= hoy]['temperature_2m_mean'].tail(7).mean()
            df_f = df[df['date'].dt.date > hoy]
            fc_t, fc_add = df_f['temperature_2m_mean'].mean(), df_f['deg_above_29'].sum()
            
            # INFERENCIA Q1 (Vuelo a 7 das)
            dist, is_disp = 19 - add29, 1 if add29 >= 19 else 0
            X = pd.DataFrame([{
                'latitud': st.session_state.lat, 'longitud': st.session_state.lon, 'altura': 450.0,
                'ADD29': add29, 'dist_to_19_gdd': dist, 'is_dispersed': is_disp,
                'tmean_roll_7': tm7, 'fcst_7d_tmean': fc_t, 'fcst_7d_ADD29_inc': fc_add
            }])
            prob = modelo.predict_proba(X)[0][1]
            
            st.divider()
            st.markdown(f"## 🔭 Informe de Riesgo (Periodo: +7 días)")
            
            # Panel de previsin
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("🎯 Probabilidad de Vuelo (Próx. 7d)")
                st.metric("Vuelo en Trampas", f"{prob:.1%}")
                if prob > 0.6:
                    st.warning("⚠️ **ALERTA DE VUELO**: Alta probabilidad de capturas la semana que viene.")
                else:
                    st.success("Baja probabilidad de actividad en este punto.")
                
            with c2:
                st.subheader("🚶 Estado de Dispersión (19 GDD)")
                st.metric("Acumulado Actual", f"{add29:.1f} GDD / 19")
                st.progress(min(1.0, add29/19))
                if add29 >= 19:
                    st.error("🚨 DISPERSI\xD3N ACTIVA: La poblaci\xf3n busca refugio.")
            
            st.divider()
            # Tendencia climtica
            st.markdown("### 🌡️ Tendencia Térmica para la semana")
            t1, t2, t3 = st.columns(3)
            t1.write(f"Previsi\xf3n Media: **{fc_t:.1f}\xbaC**")
            t2.write(f"Incremento Calor: **+{fc_add:.1f} GDD**")
            t3.write(f"Distancia Umbral 19: **{max(0, 19-add29):.1f} GDD**")
            
        except Exception as e:
            st.error("Error al obtener el pron\xf3stico a 7 d\xedas.")
