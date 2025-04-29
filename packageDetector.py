import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import autoencoder
import class_udpData

# 讀取數據
data_path = r"project\DrDoS2019_UDP_projectForm.csv"
data = class_udpData.csv_to_tensor(data_path)

# 假設數據需要標準化處理
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 將數據轉換為 PyTorch 張量
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

# 載入模型
model_path = r"project\save_model\autoencoder_model_v1.pth"
model = autoencoder.AutoEncoder(input_size=data_tensor.shape[1])  # 假設 input_size 已知
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 偵測是否異常
with torch.no_grad():
    predictions = model(data_tensor)
    # 將預測結果和輸入數據轉換為 NumPy 格式
    X_pred = predictions.cpu().numpy()
    X_np = data_tensor.cpu().numpy()

    # 計算每行的平均絕對誤差
    scored = pd.DataFrame()
    scored["Loss_mae"] = np.mean(np.abs(X_pred - X_np), axis=1)

    # 設定 MAE 閾值
    threshold = 1.65
    anomalies = scored["Loss_mae"] > threshold
    loss = scored["Loss_mae"]

# 輸出結果
data_df = pd.DataFrame(data.cpu().numpy(), columns=[f"Feature_{i}" for i in range(data.shape[1])])
data_df['Anomaly'] = anomalies.to_numpy()
data_df['Loss_mae'] = loss
for i in data_df:
    print(i, data_df[i])