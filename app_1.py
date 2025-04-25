import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_stock_data(df, lookback=100, predict_days=5):
    # 展平欄位，如果是 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel()

    # 如果欄位名稱都是股票代碼，例如 '0050.TW'，嘗試自動重命名欄位
    if all(col == df.columns[0] for col in df.columns) and isinstance(df.columns[0], str):
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # 套用標準欄名
        print("🛠 自動重命名欄位為標準格式：", df.columns)

    # 確保是 DataFrame 且有 Close 欄位
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
        return None, None, "原始資料錯誤：不是 DataFrame 或缺少 Close 欄位"

    df = df.sort_index()
    data = df[['Close']].copy()

    print("🛠 DEBUG：data 資料型態：", type(data))
    print("🛠 DEBUG：data['Close'] 類型：", type(data['Close']))
    print("🛠 DEBUG：data['Close'] 頭部內容：\n", data['Close'].head())

    if isinstance(data['Close'], pd.DataFrame):
        data['Close'] = data['Close'].squeeze()

    if data['Close'].isnull().any():
        data['Close'].fillna(method='ffill', inplace=True)

    # MACD 計算
    macd_result = ta.macd(data['Close'])
    
    print("🛠 DEBUG：macd_result 的內容：")
    print(macd_result.head())  # 顯示計算後的前幾筆資料
    print("🛠 DEBUG：macd_result 是否有 NaN：", macd_result.isna().sum())

    if macd_result is not None and not macd_result.empty:
        data['MACD'] = macd_result['MACD_12_26_9']
        data['MACD_signal'] = macd_result['MACDs_12_26_9']
    else:
        return None, None, "MACD 計算失敗，請檢查資料"

    # RSI 計算
    rsi_result = ta.rsi(data['Close'], length=14)
    if rsi_result is not None and not rsi_result.empty:
        data['RSI'] = rsi_result
    else:
        return None, None, "RSI 計算失敗，請檢查資料"

    # 去除離群值
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['Close'] >= lower_bound) & (data['Close'] <= upper_bound)]

    # 滾動平均
    data['Close'] = data['Close'].rolling(window=5).mean()

    # 去除因 MACD/RSI/rolling 產生的 NaN
    data.dropna(inplace=True)

    if data.empty:
        return None, None, "處理後資料為空"

    # 只保留模型用到的特徵
    features = ['Close', 'MACD', 'MACD_signal', 'RSI']
    data = data[features]
    data.dropna(inplace=True)

    # 標準化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 時序資料製作
    X, y = [], []
    for i in range(lookback, len(scaled_data) - predict_days + 1):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i:i + predict_days, 0])  # 只預測 Close

    return np.array(X), np.array(y), scaler

def train_lstm_model(X, y, predict_days):
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.2),
        Dense(predict_days)
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    # 使用 st.progress() 顯示訓練進度條
    progress_bar = st.progress(0)  # 初始進度為 0
    
    # 訓練模型並更新進度條
    for epoch in range(100):
        model.fit(X, y, epochs=1, batch_size=32, verbose=0,
                  validation_split=0.2, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

        # 更新進度條
        progress = (epoch + 1) / 100
        progress_bar.progress(progress)

    return model

def time_series_cross_validation(data, lookback, predict_days, n_splits=5):
    errors = []
    progress_bar = st.progress(0.0)  # 初始化進度條
    
    for split in range(n_splits):
        # ===== 進度條更新 =====
        progress_bar.progress((split + 1) / n_splits)
        
        train_size = int(len(data) * (split + 1) / n_splits)
        train_data = data[:train_size]
        
        try:
            val_data = data[train_size:train_size + predict_days, 0]  # 只看 close
        except IndexError:
            st.warning("⚠ 資料切片錯誤，跳過這一組 split")
            continue

        if len(val_data) < predict_days:
            continue

        X, y = [], []
        for i in range(lookback, len(train_data) - predict_days + 1):
            X.append(train_data[i - lookback:i])
            y.append(train_data[i:i + predict_days, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(predict_days)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, 
                  validation_split=0.2,  callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])

        val_sequence = data[train_size - lookback:train_size]
        val_sequence = np.expand_dims(val_sequence, axis=0)
        predicted = model.predict(val_sequence)
        mse = mean_squared_error(val_data, predicted.flatten())
        errors.append(mse)
        
    progress_bar.progress(1.0)  # 訓練完成，進度條設為 100%
    return np.mean(errors)

# ========== Streamlit UI ==========
st.title("📈 股票價格預測 - LSTM 模型 (含 RSI + MACD)")
option = st.selectbox("選擇操作", ("一般預測", "時間序列交叉驗證"))
ticker_input = st.text_input("輸入股票代碼（如 0050.TW）", "0050.TW")
predict_days = 5
lookback = 100

if option == "一般預測":
    if st.button("開始預測"):
        set_seed()
        with st.spinner("下載資料與預測中..."):
            df = yf.download(ticker_input, period="10y")
            if df.empty:
                st.error("找不到股票資料")
            else:
                X, y, scaler_or_msg = prepare_stock_data(df, lookback, predict_days)
                if X is None:
                    st.error(scaler_or_msg)
                else:
                    n_models = 3
                    all_predictions = []
                    
                    for i in range(n_models):
                        st.write(f"🔁 第 {i+1} 次模型訓練與預測")
                        model = train_lstm_model(X, y, predict_days)
                        latest_sequence = X[-1:]
                        predicted = model.predict(latest_sequence)
                        all_predictions.append(predicted.reshape(-1))
                    
                    # 平均預測結果
                    avg_pred = np.mean(all_predictions, axis=0)
                    
                    # 還原成價格
                    avg_prices = scaler_or_msg.inverse_transform(
                        np.hstack([avg_pred.reshape(-1, 1), np.zeros((predict_days, 3))])
                    )[:, 0]
                    
                    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_days, freq='B')
                    pred_df_avg = pd.DataFrame({'Date': future_dates, 'Price': avg_prices, 'Type': '平均預測'})
                    
                    # 每次的預測也做出 DataFrame（可選）
                    indiv_dfs = []
                    for i, pred in enumerate(all_predictions):
                        prices = scaler_or_msg.inverse_transform(
                            np.hstack([pred.reshape(-1, 1), np.zeros((predict_days, 3))])
                        )[:, 0]
                        df_pred = pd.DataFrame({'Date': future_dates, 'Price': prices, 'Type': f'第{i+1}次預測'})
                        indiv_dfs.append(df_pred)
                    
                    # 合併所有預測
                    df_plot = df[['Close']].reset_index().rename(columns={'Date': 'Date', 'Close': 'Price'})
                    df_plot['Type'] = '歷史價格'
                    combined_df = pd.concat([df_plot] + indiv_dfs + [pred_df_avg])

                    predicted_prices = scaler_or_msg.inverse_transform(
                        np.hstack([predicted.reshape(-1, 1),
                                   np.zeros((predict_days, 3))])
                    )[:, 0]  # 只取還原後的 Close
                    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_days, freq='B')
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

                    df_plot = df[['Close']].reset_index().rename(columns={'Date': 'Date', 'Close': 'Price'})
                    df_plot['Type'] = '歷史價格'
                    pred_df['Type'] = '預測價格'
                    pred_df.rename(columns={'Predicted Price': 'Price'}, inplace=True)
                    combined_df = pd.concat([df_plot, pred_df])

                    fig = go.Figure()
                    for name, group in combined_df.groupby("Type"):
                        fig.add_trace(go.Scatter(x=group["Date"], y=group["Price"],
                                                 mode="lines+markers" if name == "預測價格" else "lines",
                                                 name=name))
                    fig.update_layout(title=f"{ticker_input} 未來 {predict_days} 天預測",
                                      xaxis_title="日期", yaxis_title="股價", template="plotly_white")
                    st.plotly_chart(fig)
                    st.dataframe(pred_df.set_index('Date'))

elif option == "時間序列交叉驗證":
    if st.button("開始驗證"):
        set_seed()
        with st.spinner("執行交叉驗證中..."):
            df = yf.download(ticker_input, period="10y")
            if df.empty:
                st.error("找不到股票資料")
            else:
                X, y, scaler_or_msg = prepare_stock_data(df, lookback, predict_days)
                if X is None:
                    st.error(scaler_or_msg)
                else:
                    scaled_data = X.reshape(-1, X.shape[2])  # 還原成 2D 結構
                    mean_mse = time_series_cross_validation(scaled_data, lookback, predict_days, n_splits=3)
                    st.write(f"✅ 交叉驗證平均 MSE：{mean_mse:.4f}")
