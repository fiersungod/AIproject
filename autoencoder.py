import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import class_udpData
from sklearn.metrics import accuracy_score, f1_score

# # Generate dummy dataset (replace with real network packet data)
# def generate_dummy_data(num_samples=10000, num_features=20):
#     X = np.random.rand(num_samples, num_features)
#     y = np.random.randint(0, 2, num_samples)  # 0 for normal, 1 for abnormal
#     return X, y



# Define the AutoEncoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 定義訓練函式
def train_model(epochs=1000000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # 每 1000 次輸出一次 loss 值
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # 每隔一定的 epoch 保存模型
        if (epoch + 1) % 100000 == 0:  # 每 1000 個 epoch 保存一次
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    csv_path = r"project\20250419235916.csv"
    X = class_udpData.csv_to_tensor(csv_path)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Initialize model, loss function, and optimizer
    model = AutoEncoder(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)

    # 加載模型和優化器的狀態（如果存在）
    checkpoint_path = r'project\save_model\autoencoder_model_v1.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded. Starting new training session.")
    except FileNotFoundError:
        print("Checkpoint not found. Starting training from scratch.")

    epochs = 0
    train_model(epochs=epochs)

    # 評估模型
    with torch.no_grad():
        model.eval()  # 設置模型為評估模式
        X_test_reconstructed = model(X_test)  # 使用測試數據進行重建
        reconstruction_loss = criterion(X_test_reconstructed, X_test)  # 計算測試損失
        print(f'Test Reconstruction Loss: {reconstruction_loss.item():.4f}')

        X_pred = model(X_train).cpu().numpy()  # 將預測結果轉換為 NumPy 格式
        # 計算每行的平均絕對誤差
        X_train_np = X_train.cpu().numpy()  # 將訓練數據轉換為 NumPy 格式
        scored = pd.DataFrame()
        scored["Loss_mae"] = np.mean(np.abs(X_pred - X_train_np), axis=1)

        # 繪製分佈圖
        plt.figure()
        sns.histplot(scored["Loss_mae"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
        plt.xlim(0.0, 1.0)  # 設置 x 軸範圍
        plt.show()

        # 保存最終模型
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': reconstruction_loss.item()
        }, checkpoint_path)
        print("Final model saved.")