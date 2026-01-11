import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import asyncio
import requests
import aiohttp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

st.set_page_config(page_title="Анализ температурных данных", page_icon="thermometer", layout="wide")

st.title("Анализ температурных данных")
st.markdown("Приложение для анализа исторических температур и текущей погоды через OpenWeatherMap API.")

def calculate_rolling_stats(df_city, window=30):
    df = df_city.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['rolling_mean'] = df['temperature'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df['temperature'].rolling(window=window, center=True).std()
    df['is_anomaly'] = (
        (df['temperature'] > df['rolling_mean'] + 2 * df['rolling_std']) |
        (df['temperature'] < df['rolling_mean'] - 2 * df['rolling_std'])
    )
    return df

def analyze_city_parallel_wrapper(args):
    df, city = args
    city_df = df[df['city'] == city].copy()
    return calculate_rolling_stats(city_df)

def parallel_analysis(df, cities):
    start_time = time.time()
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=min(len(cities), 4)) as executor:
            futures = [
                executor.submit(analyze_city_parallel_wrapper, (df, city))
                for city in cities
            ]
            for i, future in enumerate(as_completed(futures)):
                city = cities[i]
                try:
                    results[city] = future.result(timeout=30)
                except Exception as exc:
                    st.error(f"Ошибка при анализе города {city}: {exc}")
    except RuntimeError as e:
        st.error(f"Ошибка выполнения параллельного анализа: {e}")
        return {}, 0
    end_time = time.time()
    duration = end_time - start_time
    return results, duration

def sequential_analysis(df, cities):
    start_time = time.time()
    for city in cities:
        city_df = df[df['city'] == city].copy()
        calculate_rolling_stats(city_df)
    end_time = time.time()
    return end_time - start_time

def get_current_weather_sync(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

async def get_current_weather_async(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}

def validate_api_key(api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": "London", "appid": api_key}
    try:
        response = requests.get(url, params=params, timeout=5)
        return response.status_code == 200
    except:
        return False

def get_season_from_date(date_str):
    month = int(pd.to_datetime(date_str).month)
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"

def check_temperature_anomaly(current_temp, city, season, df_hist):
    hist_data = df_hist[(df_hist['city'] == city) & (df_hist['season'] == season)]
    if hist_data.empty:
        return False
    mean_temp = hist_data['temperature'].mean()
    std_temp = hist_data['temperature'].std()
    lower_bound = mean_temp - 2 * std_temp
    upper_bound = mean_temp + 2 * std_temp
    return not (lower_bound <= current_temp <= upper_bound)

st.subheader("Загрузка данных")
use_preloaded = st.checkbox(
    "Использовать предзагруженные данные",
    value=True,
    help="Данные включают 5 лет наблюдений для 5 городов."
)

df = None
if use_preloaded:
    data_path = "data/temperature_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.success("Данные загружены")
    else:
        st.error(f"Файл не найден: {data_path}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Файл загружен")
    else:
        st.info("Загрузите CSV-файл или используйте предзагруженные данные.")
        st.stop()

required_columns = {"city", "timestamp", "temperature", "season"}
if not required_columns.issubset(df.columns):
    st.error(f"Файл должен содержать колонки: {required_columns}")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])

st.subheader("Выбор города")
cities = sorted(df["city"].unique())
selected_city = st.selectbox("Город", cities)

if selected_city in ["Moscow", "Beijing"]:
    st.warning(f"В городе {selected_city} искусственно введены аномалии.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Анализ", "API", "Параллелизм", "Экспорт", "Карта", "Сравнение"
])

with tab1:
    st.subheader("Анализ температурных данных")
    city_data = df[df["city"] == selected_city].copy()
    if len(city_data) == 0:
        st.warning(f"Нет данных для {selected_city}")
    else:
        st.write(f"Записей: {len(city_data)}")
        if st.button("Запустить анализ"):
            analyzed_city = calculate_rolling_stats(city_data)
            city_data = analyzed_city
            anomalies_count = city_data['is_anomaly'].sum()
            st.metric("Аномалий", anomalies_count)
            
            st.subheader("Статистика")
            desc_stats = city_data[['temperature']].describe()
            st.dataframe(desc_stats, width='stretch')

            st.subheader("График температуры")
            fig = px.line(
                city_data,
                x='timestamp',
                y='temperature',
                title=f'Температура в {selected_city}',
                line_shape='spline'
            )

            anomalies = city_data[city_data['is_anomaly']]
            fig.add_scatter(
                x=anomalies['timestamp'],
                y=anomalies['temperature'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Аномалии'
            )

            fig.add_trace(
                go.Scatter(
                    x=city_data['timestamp'],
                    y=city_data['rolling_mean'],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Скользящее среднее'
                )
            )

            x_numeric = list(range(len(city_data)))
            y_values = city_data['temperature'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
            trend_line = slope * np.array(x_numeric) + intercept

            fig.add_trace(
                go.Scatter(
                    x=city_data['timestamp'],
                    y=trend_line,
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name='Тренд'
                )
            )

            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            seasonal_data = city_data.groupby('season').agg({
                'temperature': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            seasonal_data.columns = ['Средняя', 'STD', 'Мин', 'Макс', 'Количество']
            st.dataframe(seasonal_data, width='stretch')

            if anomalies_count > 0:
                st.subheader("Первые аномалии")
                st.dataframe(city_data[city_data['is_anomaly']].head(5), width='stretch')

with tab2:
    st.subheader("Текущая погода")
    api_key = st.text_input("API-ключ OpenWeatherMap", type="password")
    if api_key:
        if validate_api_key(api_key):
            st.success("Ключ действителен")
        else:
            st.error("Неверный ключ")
            st.stop()

        if st.button("Получить погоду"):
            start_time_sync = time.time()
            weather_sync = get_current_weather_sync(selected_city, api_key)
            end_time_sync = time.time()

            start_time_async = time.time()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            weather_async = loop.run_until_complete(get_current_weather_async(selected_city, api_key))
            loop.close()
            end_time_async = time.time()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Время синхронного запроса", f"{end_time_sync - start_time_sync:.3f} сек")
            with col2:
                st.metric("Время асинхронного запроса", f"{end_time_async - start_time_async:.3f} сек")

            if 'main' in weather_sync:
                current_temp = weather_sync['main']['temp']
                current_desc = weather_sync['weather'][0]['description']
                today_season = get_season_from_date(pd.Timestamp.now().strftime("%Y-%m-%d"))
                is_anomaly_now = check_temperature_anomaly(current_temp, selected_city, today_season, df)

                st.subheader(f"Погода в {selected_city} сейчас:")
                st.metric("Температура", f"{current_temp} °C")
                st.write(f"Описание: {current_desc.capitalize()}")

                if is_anomaly_now:
                    st.error(f"Температура аномальна для {today_season} в {selected_city}!")
                else:
                    st.success(f"Температура в пределах нормы для {today_season}.")
            else:
                st.error(f"Ошибка от API: {weather_sync}")

with tab3:
    st.subheader("Параллельный анализ")
    if st.button("Сравнить время"):
        all_cities = df['city'].unique().tolist()
        with st.spinner("Анализ..."):
            seq_time = sequential_analysis(df, all_cities)
            _, par_time = parallel_analysis(df, all_cities)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Последовательный", f"{seq_time:.2f} сек")
            with col2:
                st.metric("Параллельный", f"{par_time:.2f} сек")
            if par_time > 0:
                with col3:
                    st.metric("Ускорение", f"{seq_time / par_time:.2f}x")

with tab4:
    st.subheader("Экспорт данных")
    if 'city_data' in locals():
        st.success("Данные готовы к экспорту.")
        st.download_button(
            label="Скачать результаты",
            data=city_data.to_csv(index=False).encode('utf-8'),
            file_name=f"results_{selected_city}.csv",
            mime="text/csv"
        )
    else:
        st.info("Проведите анализ, чтобы экспортировать данные.")

with tab5:
    st.subheader("Карта погоды")
    if api_key:
        if st.button("Показать на карте"):
            weather = get_current_weather_sync(selected_city, api_key)
            if 'coord' in weather and 'main' in weather:
                lat = weather['coord']['lat']
                lon = weather['coord']['lon']
                temp = weather['main']['temp']
                fig_map = go.Figure(data=go.Scattergeo(
                    lon=[lon],
                    lat=[lat],
                    text=[f"{selected_city}<br>{temp}°C"],
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=0.8),
                    hovertemplate='%{text}<extra></extra>'
                ))
                fig_map.update_geos(projection_type="natural earth")
                fig_map.update_layout(
                    title=f"Текущая температура в {selected_city}",
                    geo=dict(showland=True, landcolor="lightgray"),
                    height=500
                )
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.error(f"Ошибка от API: {weather}")

with tab6:
    st.subheader("Сравнение городов")
    city1 = st.selectbox("Первый город", cities, index=0)
    city2 = st.selectbox("Второй город", cities, index=1)
    if st.button("Сравнить"):
        data1 = df[df['city'] == city1]
        data2 = df[df['city'] == city2]
        if len(data1) > 0 and len(data2) > 0:
            fig_comp = px.line(title=f"Сравнение температур: {city1} vs {city2}")
            fig_comp.add_scatter(x=data1['timestamp'], y=data1['temperature'], name=city1, mode='lines')
            fig_comp.add_scatter(x=data2['timestamp'], y=data2['temperature'], name=city2, mode='lines')
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Нет данных для одного или обоих городов.")

with st.sidebar.expander("ℹ️ О проекте"):
    st.markdown("""
    ### Функционал приложения:

    - **Загрузка данных**: CSV или предзагруженный файл
    - **Анализ температур**: Скользящее среднее, аномалии, тренды
    - **Сезонные профили**: Графики и статистика по сезонам
    - **API OpenWeatherMap**: Текущая погода, сравнение с историей
    - **Параллельный анализ**: Сравнение времени выполнения
    - **Экспорт CSV**: Результаты анализа
    - **Карта погоды**: Визуализация на карте
    - **Сравнение городов**: График температур
    """)

if st.sidebar.button("Сбросить"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()