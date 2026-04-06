import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import requests
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Prediccin Mosca V20 - Final", page_icon="🪰", layout="wide")

st.title("🪰 Monitor de IA Mosca (V20 - Versión Final)")
st.markdown("Diagnstico de **Presencia**, **Predicción** y **Contador de Dispersión** (Filtro Media > 29ºC).")

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
    attr='Google', name='Sat\xe9lite H\xedbrido', overlay=False, control=True
).add_to(m)
folium.Marker(location=[st.session_state.lat, st.session_state.lon], icon=folium.Icon(color='red')).add_to(m)

map_out = st_folium(m, width=950, height=400, key="mapa_v20")

if map_out and map_out.get("last_clicked"):
    nl, ng = map_out["last_clicked"]["lat"], map_out["last_clicked"]["lng"]
    if nl != st.session_state.lat:
        st.session_state.lat, st.session_state.lon = nl, ng
        st.rerun()

st.sidebar.header("🕹️ Centro de Control")
st.sidebar.info(f"📍 Coords: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

# --- DESCARGA ---
def get_v20_data(lat, lon):
    hoy = datetime.date.today()
    inicio = datetime.date(hoy.year, 1, 1)
    url_f = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=92&forecast_days=7&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    url_a = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={inicio}&end_date={hoy - datetime.timedelta(days=10)}&daily=temperature_2m_max,temperature_2m_mean&timezone=Europe/Madrid"
    
    try:
        r_f = requests.get(url_f).json()['daily']
        df = pd.DataFrame(r_f)
        try:
            r_a = requests.get(url_a).json()['daily']
            df_a = pd.DataFrame(r_a)
            df = pd.concat([df_a, df]).drop_duplicates(subset=['time']).sort_values('time')
        except: pass
            
        df['date'] = pd.to_datetime(df['time']).dt.date
        
        # --- LÓGICA CIENTÍFICA V20: Solo si TMedia > 29 ---
        df['deg_above_29_raw'] = np.maximum(0, df['temperature_2m_max'] - 29)
        df['deg_above_29'] = np.where(df['temperature_2m_mean'] > 29, df['deg_above_29_raw'], 0)
        
        df['ADD29'] = df['deg_above_29'].cumsum()
        
        # OBTENEMOS ALTITUD REAL DE LA API
        elevacion = requests.get(url_f).json().get('elevation', 450.0)
        
        return df, elevacion
    except: return None, 450.0

if st.sidebar.button("🚀 Lanzar Análisis Completo", type="primary"):
    with st.spinner("Analizando clima y biología..."):
        df, alt_real = get_v20_data(st.session_state.lat, st.session_state.lon)
        
        if df is not None:
            st.sidebar.success(f"📏 Altitud detectada: {alt_real} m")
            hoy = datetime.date.today()
            actual = df[df['date'] <= hoy].iloc[-1]
            add29 = actual['ADD29']
            tm7_act = df[df['date'] <= hoy]['temperature_2m_mean'].tail(7).mean()
            
            # FUTURO
            fut = df[df['date'] > hoy]
            fc_t_media, fc_add = fut['temperature_2m_mean'].mean(), fut['deg_above_29'].sum()
            
            # IA
            d19, is_d = 19 - add29, 1 if add29 >= 19 else 0
            X = pd.DataFrame([{
                'latitud': st.session_state.lat, 'longitud': st.session_state.lon, 'altura': float(alt_real),
                'ADD29': add29, 'dist_to_19_gdd': d19, 'is_dispersed': is_d,
                'tmean_roll_7': tm7_act, 'fcst_7d_tmean': fc_t_media, 'fcst_7d_ADD29_inc': fc_add
            }])
            prob = modelo.predict_proba(X)[0][1]
            
            # --- INTERFAZ ---
            st.divider()
            st.write(f"## 🔭 Informe para ({st.session_state.lat:.4f}, {st.session_state.lon:.4f})")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📍 ESTADO ACTUAL (HOY)")
                if add29 >= 19.0:
                    st.error("🔴 AUSENCIA POR DISPERSIÓN")
                elif prob > 0.5:
                    st.success("🟢 PRESENCIA ACTIVA")
                else:
                    st.info("⚪ AUSENCIA (Baja Actividad)")
                
                # CONTADOR GDD
                sc1, sc2 = st.columns([1.2, 1])
                sc1.metric("Contador Acumulado", f"{add29:.1f} GDD")
                sc2.metric("Umbral Crítico", "19.0 GDD")
                st.progress(min(1.0, add29/19.0))
                if add29 >= 19:
                    st.error("🚨 DISPERSIÓN ACTIVA: Umbral de 19 superado.")
                elif add29 >= 14:
                    st.warning(f"⚠️ Alerta: Faltan solo **{19-add29:.1f} GDD** para la dispersión.")

            with c2:
                st.subheader("🔮 PREDICCIÓN (PRÓX. SEMANA)")
                st.markdown(f"""
                    <div style="background-color:#fef9e7; padding:20px; border-radius:10px; border-left: 10px solid #f1c40f;">
                        <h1 style="margin:0; color:#2c3e50; font-size: 50px;">{prob:.1%}<span style="font-size:20px;"> Prob. Presencia</span></h1>
                        <p style="margin:0; font-weight:bold;">Estado previsto para dentro de 7 días.</p>
                    </div>
                """, unsafe_allow_html=True)
                if fc_t_media > 29.0:
                    st.warning(f"🔥 Carga térmica prevista semanal: **+{fc_add:.1f} GDD** adicionales (TMedia > 29ºC).")
                else:
                    st.info(f"ℹ️ Sin carga t\xe9rmica prevista (TMedia semanal: **{fc_t_media:.1f} \xbaC**).")

            st.divider()
            st.subheader("🗓️ Detalle de Previsión Térmica Diaria")
            tabla_f = fut[['date', 'temperature_2m_mean', 'temperature_2m_max']].copy()
            tabla_f.columns = ['Fecha', 'Media (ºC)', 'Máxima (ºC)']
            tabla_f['Carga Térmica (>29ºC Media)'] = tabla_f['Media (ºC)'].apply(lambda x: "🚩 Sí" if x > 29 else "Nula")
            st.table(tabla_f.set_index('Fecha'))
            
        else:
            st.error("Error al obtener los datos del satélite.")
