import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import autoencoder
import class_udpData
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取數據
data_path = r"project\20250417140400-39.csv"
data = class_udpData.csv_to_tensor(data_path)

# 假設數據需要標準化處理
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 將數據轉換為 PyTorch 張量並移動到指定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

# 載入模型
model_path = r"project\save_model\autoencoder_model_broken.pth"
model = autoencoder.AutoEncoder(input_size=data_tensor.shape[1]).to(device)  # 將模型移動到設備
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 偵測是否異常
with torch.no_grad():
    predictions = model(data_tensor)
    # 將預測結果和輸入數據轉換為 NumPy 格式
    X_pred = predictions.to("cpu").numpy()  # 從設備移回 CPU 並轉換為 NumPy
    X_np = data_tensor.to("cpu").numpy()  # 從設備移回 CPU 並轉換為 NumPy

    # 計算每行的平均絕對誤差
    scored = pd.DataFrame()
    scored["Loss_mae"] = np.mean(np.abs(X_pred - X_np), axis=1)

    # 設定 MAE 閾值
    threshold = 1.5
    anomalies = scored["Loss_mae"] > threshold
    loss = scored["Loss_mae"]

# 將數據轉換為 pandas.DataFrame 並移回 CPU
data_df = pd.DataFrame(data_tensor.to("cpu").numpy(), columns=[f"Feature_{i}" for i in range(data_tensor.shape[1])])
data_df['Anomaly'] = anomalies.to_numpy()  # 將異常標記添加到 DataFrame
data_df['Loss_mae'] = loss  # 將損失值添加到 DataFrame

for i in range(len(data_df['Anomaly'])):
    if data_df['Anomaly'][i]:
        print(data_df["Anomaly"][i], data_df["Loss_mae"][i]) #data_df['Loss_mae'][i])

plt.figure()
sns.histplot(loss, bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
plt.xlim(0.0, 5.0)  # 設置 x 軸範圍
plt.show()