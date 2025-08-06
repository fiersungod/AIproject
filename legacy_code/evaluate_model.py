import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import class_udpData
import os
import autoencoder

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加載數據
csv_path = r"project\20250502150415.csv"
X = class_udpData.csv_to_tensor(csv_path)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# 初始化模型
model = autoencoder.AutoEncoder(input_size=X_tensor.shape[1]).to(device)

# 加載模型權重
checkpoint_path = r'project\save_model\autoencoder_model_notgood.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully.")
except FileNotFoundError:
    print("Checkpoint not found. Please ensure the model file exists.")
    exit()

# 使用模型進行預測
model.eval()
with torch.no_grad():
    X_pred = model(X_tensor).cpu().numpy()  # 將預測結果轉換為 NumPy 格式
    X_np = X_tensor.cpu().numpy()  # 將輸入數據轉換為 NumPy 格式

# 計算每行的平均絕對誤差
scored = pd.DataFrame()
scored["Loss_mae"] = np.mean(np.abs(X_pred - X_np), axis=1)

# 繪製分佈圖
plt.figure()
sns.histplot(scored["Loss_mae"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
plt.xlim(0.0, 5.0)  # 設置 x 軸範圍
plt.show()