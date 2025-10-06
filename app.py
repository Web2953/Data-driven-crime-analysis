
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_auc_score)
from prophet import Prophet
from prophet.plot import plot_plotly
import geopandas as gpd
from shapely.geometry import Point, box
import folium
from streamlit_folium import st_folium
import math
import datetime
from functools import lru_cache

# -------------------------
# CONFIG / DATA PATHS
# -------------------------
st.set_page_config(layout="wide", page_title="Crime Hotspots Dashboard")

# Replace these with your actual datasets (local path or direct URL if accessible)
DATA_CRIME_CSV = "data/crime_incidents.csv"       # main crimes table: columns expected: ['station_id','station_name','province','category','count','year','month','lat','lon']
DATA_STATIONS_CSV = "data/police_stations.csv"    # station metadata: ['station_id','name','lat','lon','num_officers','station_size','budget_allocation']
DATA_DEMOGRAPHICS_CSV = "data/demographics.csv"  # supporting contextual data keyed by station or municipality

# Quick UI header
st.title("Crime Hotspots — Classification, Forecasting & Drone Simulation")
st.write("Interactive app for analysis, modeling and a simple drone simulation to visit identified hotspots.")

# -------------------------
# DATA LOADING / PREPROCESS
# -------------------------
@st.cache_data(ttl=3600)
def load_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Unable to load {path}: {e}")
        return pd.DataFrame()

crime_df = load_csv(DATA_CRIME_CSV)
stations_df = load_csv(DATA_STATIONS_CSV)
demo_df = load_csv(DATA_DEMOGRAPHICS_CSV)

# If datasets did not load, show helpful message + example schema
if crime_df.empty:
    st.warning("crime_incidents.csv not found or empty. The app UI will still show structure. Replace DATA_CRIME_CSV with an actual path.")
    st.markdown("**Expected crime_incidents.csv columns:** `station_id, station_name, province, category, count, year, month, lat, lon`")
if stations_df.empty:
    st.warning("police_stations.csv not found or empty. Replace DATA_STATIONS_CSV with actual station metadata.")
    st.markdown("**Expected police_stations.csv columns:** `station_id, name, lat, lon, num_officers, station_size, budget_allocation`")

# Basic preprocessing if data exists
if not crime_df.empty:
    # Standardize column names
    crime_df.columns = [c.strip() for c in crime_df.columns]
    if 'date' in crime_df.columns:
        crime_df['date'] = pd.to_datetime(crime_df['date'])
        crime_df['year'] = crime_df['date'].dt.year
        crime_df['month'] = crime_df['date'].dt.month
    else:
        # If only year/month exist, make a synthetic date as first day of month
        if 'year' in crime_df.columns and 'month' in crime_df.columns:
            crime_df['date'] = pd.to_datetime(crime_df[['year','month']].assign(DAY=1))
        elif 'year' in crime_df.columns:
            crime_df['date'] = pd.to_datetime(crime_df['year'].astype(str) + "-01-01")
    # Ensure lat/lon exist; try to merge from stations table if missing
    if ('lat' not in crime_df.columns or 'lon' not in crime_df.columns) and not stations_df.empty:
        if 'station_id' in crime_df.columns and 'station_id' in stations_df.columns:
            stations_df = stations_df.rename(columns=lambda x: x.strip())
            crime_df = crime_df.merge(stations_df[['station_id','lat','lon']], on='station_id', how='left')
    # fill na counts
    if 'count' in crime_df.columns:
        crime_df['count'] = crime_df['count'].fillna(0).astype(int)

# Merge crime + stations + demographics for features (if available)
@st.cache_data(ttl=3600)
def build_master(crime, stations, demo):
    if crime.empty:
        return pd.DataFrame()
    df = crime.copy()
    if not stations.empty and 'station_id' in df.columns:
        df = df.merge(stations, on='station_id', how='left', suffixes=('','_stat'))
    if not demo.empty:
        # assume demo keyed by station_id or municipality_name
        if 'station_id' in demo.columns and 'station_id' in df.columns:
            df = df.merge(demo, on='station_id', how='left')
    return df

master_df = build_master(crime_df, stations_df, demo_df)

# -------------------------
# SIDEBAR: FILTERS
# -------------------------
st.sidebar.header("Filters")
# Category filter
if 'category' in master_df.columns:
    categories = sorted(master_df['category'].dropna().unique())
    sel_categories = st.sidebar.multiselect("Crime categories", options=categories, default=categories[:3])
else:
    sel_categories = []

# Location filter (province or station)
location_field = None
if 'province' in master_df.columns:
    provinces = sorted(master_df['province'].dropna().unique())
    sel_provinces = st.sidebar.multiselect("Province", options=provinces, default=provinces[:3])
    location_field = 'province'
else:
    sel_provinces = []

# Date filter
if 'date' in master_df.columns and not master_df.empty:
    min_date = master_df['date'].min()
    max_date = master_df['date'].max()
else:
    min_date = pd.to_datetime("2011-01-01")
    max_date = pd.to_datetime("2023-12-31")

start_date, end_date = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Agg granularity
agg_grain = st.sidebar.selectbox("Aggregation level", ['monthly', 'quarterly', 'annual'])

# Hotspot definition params
st.sidebar.header("Hotspot definition")
top_n = st.sidebar.number_input("Top N precincts = hotspot", min_value=5, max_value=200, value=25, step=5)

# Forecast horizon
st.sidebar.header("Forecast")
forecast_horizon_months = st.sidebar.slider("Forecast horizon (months)", 12, 24, 12)

# Train ML?
do_train = st.sidebar.checkbox("Train classification model now", value=True)

# -------------------------
# DATA SUBSET BASED ON FILTERS
# -------------------------
def subset_data(df):
    if df.empty:
        return df
    sub = df.copy()
    if sel_categories:
        sub = sub[sub['category'].isin(sel_categories)]
    if sel_provinces:
        sub = sub[sub['province'].isin(sel_provinces)]
    sub = sub[(sub['date'] >= pd.to_datetime(start_date)) & (sub['date'] <= pd.to_datetime(end_date))]
    return sub

subset = subset_data(master_df)
st.markdown(f"### Data snapshot ({len(subset)} rows)")
st.dataframe(subset.head(10))

# -------------------------
# EDA: Time series, Map, Distribution
# -------------------------
st.markdown("## Exploratory Data Analysis (EDA)")
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Time series: selected categories aggregated by month")
    if subset.empty:
        st.info("No data to plot. Load your CSV files into the DATA_* paths.")
    else:
        # aggregate monthly
        ts = subset.groupby([pd.Grouper(key='date', freq='M')])['count'].sum().reset_index()
        fig = px.line(ts, x='date', y='count', title='Total crimes over time (selected filters)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Heatmap by station (counts)")
    if 'lat' in subset.columns and 'lon' in subset.columns:
        station_sum = subset.groupby(['station_id','station_name','lat','lon'])['count'].sum().reset_index()
        if not station_sum.empty:
            m = folium.Map(location=[station_sum['lat'].mean(), station_sum['lon'].mean()], zoom_start=7)
            for _, r in station_sum.iterrows():
                folium.CircleMarker(
                    location=[r['lat'], r['lon']],
                    radius=3 + math.log1p(r['count']),
                    popup=f"{r.get('station_name','')}: {r['count']}",
                ).add_to(m)
            st_folium(m, width=700, height=400)
        else:
            st.info("No station coordinates found for mapped heatmap.")
    else:
        st.info("Latitude/Longitude not available - load station metadata with lat/lon.")

with col2:
    st.markdown("### Distribution by category")
    if not subset.empty and 'category' in subset.columns:
        cat_sum = subset.groupby('category')['count'].sum().reset_index().sort_values('count',ascending=False)
        fig2 = px.bar(cat_sum, x='category', y='count', title='Counts by category')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No category data available.")

# -------------------------
# HOTSPOT LABELS
# -------------------------
st.markdown("## Hotspot labeling")
def label_hotspots(df, top_n):
    if df.empty:
        return df
    total_by_station = df.groupby('station_id', as_index=False)['count'].sum()
    total_by_station = total_by_station.sort_values('count', ascending=False)
    top = total_by_station.head(top_n)['station_id'].tolist()
    # create mapping
    df['hotspot'] = df['station_id'].apply(lambda x: 1 if x in top else 0)
    return df, total_by_station

labeled_df, station_totals = label_hotspots(subset.copy(), top_n)
st.dataframe(station_totals.head(10))

st.markdown("**Hotspot rule:** top-{} stations by total counts in the selected window.".format(top_n))

# -------------------------
# CLASSIFICATION MODEL
# -------------------------
st.markdown("## Classification: predict Hotspot (1) vs Not (0)")
if labeled_df.empty:
    st.info("No data for classification (ensure station_id and counts are present).")
else:
    # Build feature set
    # Features: Crime Type (one-hot), Crime Count (lagged aggregated), Year, lat/lon, num_officers, station_size, budget_allocation
    df = labeled_df.copy()
    # aggregate per station-time-window (e.g., monthly)
    agg_cols = ['station_id','date','hotspot','lat','lon','num_officers','station_size','budget_allocation','category','count']
    df_feat = df[agg_cols].copy()
    # pivot categories to one-hot features per station-date
    df_pivot = df_feat.pivot_table(index=['station_id','date','lat','lon','num_officers','station_size','budget_allocation','hotspot'],
                                   columns='category', values='count', aggfunc='sum', fill_value=0).reset_index()
    # Add temporal features
    df_pivot['year'] = df_pivot['date'].dt.year
    df_pivot['month'] = df_pivot['date'].dt.month
    # Fill officers/budget with median if missing
    for c in ['num_officers','station_size','budget_allocation']:
        if c in df_pivot.columns:
            df_pivot[c] = df_pivot[c].fillna(df_pivot[c].median())
    # Prepare X,y
    X = df_pivot.drop(columns=['station_id','date','hotspot'])
    y = df_pivot['hotspot'].astype(int)

    # train/test split: use time-aware split
    # sort by date then split last 20% as test
    df_pivot = df_pivot.sort_values('date')
    split_idx = int(len(df_pivot)*0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    if do_train:
        st.markdown("### Training RandomForestClassifier (cached)")
        @st.cache_resource
        def train_rf(X_tr, y_tr):
            rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            rf.fit(X_tr, y_tr)
            return rf
        model = train_rf(X_train.fillna(0), y_train)
        y_pred = model.predict(X_test.fillna(0))
        y_proba = model.predict_proba(X_test.fillna(0))[:,1]

        # Metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = None

        st.write("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}, AUC: {}".format(precision, recall, f1, auc))
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.markdown("#### Classification Report (detailed)")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # Show feature importances
        feat_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
        st.markdown("#### Top feature importances")
        st.bar_chart(feat_imp)

    else:
        st.info("Training disabled. Toggle 'Train classification model now' in the sidebar to train.")

# -------------------------
# FORECASTING (Prophet)
# -------------------------
st.markdown("## Forecasting (Prophet)")
st.markdown("Select one crime category and a geographic unit (station or province) for forecasting.")

# Forecast selectors
if not subset.empty and 'category' in subset.columns:
    forecast_category = st.selectbox("Forecast category", options=sorted(subset['category'].unique()))
else:
    forecast_category = None

geo_unit = 'station_id' if 'station_id' in subset.columns else 'province' if 'province' in subset.columns else None
if geo_unit is not None:
    unique_geo = subset[geo_unit].dropna().unique()
    sel_geo = st.selectbox(f"Geographic unit ({geo_unit})", options=sorted(unique_geo))
else:
    sel_geo = None

if forecast_category and sel_geo:
    # Build monthly series
    ser = subset[(subset['category']==forecast_category) & (subset[geo_unit]==sel_geo)].groupby(pd.Grouper(key='date',freq='M'))['count'].sum().reset_index()
    ser = ser.rename(columns={'date':'ds','count':'y'})
    if len(ser) < 12:
        st.warning("Not enough data points to forecast (need at least ~12 months).")
    else:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(ser)
        future = m.make_future_dataframe(periods=forecast_horizon_months, freq='M')
        fcst = m.predict(future)
        fig = plot_plotly(m, fcst)
        st.plotly_chart(fig, use_container_width=True)
        # Show last rows of forecast
        st.dataframe(fcst[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon_months))

        # Simple non-technical summary
        last_pred = fcst[['ds','yhat']].tail(forecast_horizon_months)
        pct_change = (last_pred['yhat'].iloc[-1] - ser['y'].iloc[-1]) / max(1, ser['y'].iloc[-1]) * 100
        st.markdown("**Non-technical summary:**")
        st.write(f"Predicted {forecast_category} at {sel_geo}: expected change over the next {forecast_horizon_months} months ≈ {pct_change:.1f}% (relative to last observed month).")
        st.markdown("**Technical summary:**")
        st.write("Model used: Prophet (additive with yearly seasonality). Forecast includes 80%+95% intervals from Prophet's uncertainty model.")

# -------------------------
# DRONE SIMULATION
# -------------------------
st.markdown("## Drone Simulation: visiting hotspots")
st.markdown("This simulation creates a grid (a 3D 'frame' with height levels optional), assigns hotspots as POIs, and plans a simple route (greedy nearest neighbor).")

# Drone sim inputs
drone_start_lat = st.number_input("Drone start latitude", value=float(master_df['lat'].mean()) if 'lat' in master_df.columns else -29.0)
drone_start_lon = st.number_input("Drone start longitude", value=float(master_df['lon'].mean()) if 'lon' in master_df.columns else 24.0)
grid_cell_size_km = st.slider("Grid cell size (km)", 0.5, 10.0, 2.0, step=0.5)
altitude_levels = st.slider("3D altitude levels (number of z-layers)", 1, 5, 1)

def km_to_deg_lat(km):
    return km / 110.574

def km_to_deg_lon(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)))

def build_grid(bounds, cell_km, layers=1):
    lat_min, lat_max, lon_min, lon_max = bounds
    dlat = km_to_deg_lat(cell_km)
    dlon = km_to_deg_lon(cell_km, (lat_min+lat_max)/2)
    rows = int(np.ceil((lat_max-lat_min)/dlat))
    cols = int(np.ceil((lon_max-lon_min)/dlon))
    cells = []
    for i in range(rows):
        for j in range(cols):
            cell_box = (lat_min + i*dlat, lat_min + (i+1)*dlat, lon_min + j*dlon, lon_min + (j+1)*dlon)
            for layer in range(layers):
                cells.append({'i':i,'j':j,'layer':layer,'box':cell_box})
    return cells

# Boundaries based on current subset stations
if not subset.empty and 'lat' in subset.columns and 'lon' in subset.columns:
    lat_min, lat_max = float(subset['lat'].min()), float(subset['lat'].max())
    lon_min, lon_max = float(subset['lon'].min()), float(subset['lon'].max())
else:
    # default small box around start
    lat_min, lat_max = drone_start_lat - 0.05, drone_start_lat + 0.05
    lon_min, lon_max = drone_start_lon - 0.05, drone_start_lon + 0.05

cells = build_grid((lat_min, lat_max, lon_min, lon_max), grid_cell_size_km, altitude_levels)

# Identify hotspots in this subset (station centroids)
hotspots = station_totals.head(top_n).merge(stations_df, on='station_id', how='left') if not station_totals.empty and not stations_df.empty else pd.DataFrame()
if not hotspots.empty:
    st.markdown(f"Identified {len(hotspots)} hotspots (top {top_n}).")
    # simulate greedy route starting from drone start
    def distance(a,b):
        # haversine approx
        lat1,lon1 = a
        lat2,lon2 = b
        R=6371.0
        phi1,phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2-lat1)
        dlambda = math.radians(lon2-lon1)
        a_hav = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2*math.atan2(math.sqrt(a_hav), math.sqrt(1-a_hav))
        return R*c

    points = []
    for _, r in hotspots.iterrows():
        if pd.notnull(r.get('lat')) and pd.notnull(r.get('lon')):
            points.append((r['station_id'], r['station_name'], float(r['lat']), float(r['lon']), int(r['count'])))
    # greedy TSP
    route = []
    current = (drone_start_lat, drone_start_lon)
    remaining = points.copy()
    while remaining:
        dists = [(p, distance(current,(p[2],p[3]))) for p in remaining]
        nxt = min(dists, key=lambda x: x[1])[0]
        route.append(nxt)
        current = (nxt[2], nxt[3])
        remaining = [r for r in remaining if r[0] != nxt[0]]

    # Show route on map
    m2 = folium.Map(location=[drone_start_lat, drone_start_lon], zoom_start=10)
    folium.Marker(location=[drone_start_lat, drone_start_lon], popup="Drone start", icon=folium.Icon(color='green')).add_to(m2)
    prev = (drone_start_lat, drone_start_lon)
    for idx, p in enumerate(route):
        folium.Marker(location=[p[2], p[3]], popup=f"{p[1]} (count={p[4]})", icon=folium.Icon(color='red')).add_to(m2)
        folium.PolyLine(locations=[prev, (p[2],p[3])], color='blue').add_to(m2)
        prev = (p[2],p[3])
    st_folium(m2, width=700, height=400)

    # Print route table and ETAs (assume constant speed)
    speed_kmh = st.number_input("Drone cruise speed (km/h)", value=60.0)
    times = []
    cur = (drone_start_lat, drone_start_lon)
    total_time = 0.0
    for p in route:
        d_km = distance(cur, (p[2],p[3]))
        t_h = d_km / speed_kmh
        total_time += t_h
        times.append({'station_id':p[0],'station_name':p[1],'lat':p[2],'lon':p[3],'dist_km':round(d_km,2),'eta_hours_from_start':round(total_time,3)})
        cur = (p[2],p[3])
    st.dataframe(pd.DataFrame(times))
    st.write(f"Estimated total flight time: {total_time:.2f} hours (one-way)")

else:
    st.info("No hotspot coordinates available for drone simulation. Ensure you loaded police_stations.csv with lat/lon and that hotspot rule identified top stations.")

# -------------------------
# SUMMARY SECTIONS (FOR TECHNICAL + NON-TECH USERS)
# -------------------------
st.markdown("## Summaries")

with st.expander("Non-Technical Summary (for public/policymakers)"):
    st.write("""
    - We identified the top hotspots (top-{top}) in the selected time period and location.
    - A machine learning classifier was trained to predict if a station/time-window is a hotspot using recent crime counts, station resources (officers, size, budget) and location.
    - Forecasts (Prophet) predict short-term increases or decreases per category — use these to allocate patrols during predicted peaks.
    - A simulated drone route was generated to visit hotspots; expected flight time and route are provided as a planning aid.
    """.replace("{top}", str(top_n)))

with st.expander("Technical Summary (for analysts)"):
    st.write("""
    - Classification: RandomForestClassifier (n=200, class_weight=balanced). Features include one-hot category counts per station-time, temporal features, and station resource covariates. Evaluation: precision/recall/F1/AUC reported; confusion matrix plotted.
    - Forecasting: Prophet with yearly seasonality; forecast horizon set by user. Backtests should use rolling-origin CV for production.
    - Drone sim: grid created by converting km cell-size to degrees; hotspots assigned by top-N rule; path planned via greedy nearest-neighbor TSP heuristic. For production, use optimization (ILP/TSP solver) + no-fly zones constraint + battery model + multiple drones scheduling.
    - Caveats: models trained on reported crimes only (underreporting bias), station boundary changes over time, and data completeness must be validated.
    """)

st.markdown("### Export")
if st.button("Export hotspot list as CSV"):
    if not station_totals.empty:
        station_totals.to_csv("hotspots_topN.csv", index=False)
        st.success("hotspots_topN.csv exported. Check your working directory.")
    else:
        st.info("No hotspots to export.")

st.markdown("App created by: Your Name — adapt dataset paths at the top of this script (DATA_CRIME_CSV, DATA_STATIONS_CSV, DATA_DEMOGRAPHICS_CSV).")
