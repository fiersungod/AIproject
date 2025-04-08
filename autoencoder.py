import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import class_udpData

# # Generate dummy dataset (replace with real network packet data)
# def generate_dummy_data(num_samples=10000, num_features=20):
#     X = np.random.rand(num_samples, num_features)
#     y = np.random.randint(0, 2, num_samples)  # 0 for normal, 1 for abnormal
#     return X, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the AutoEncoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
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

# Load dataset
csv_path = r"D:\Code\project\20250331134719.csv"
X = class_udpData.csv_to_tensor(csv_path)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

# Initialize model, loss function, and optimizer
model = AutoEncoder(input_size=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def train_model(epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Training loop
epochs = 1000
for epoch in range(epochs):
    train_model()

# Evaluate model
with torch.no_grad():
    X_test_reconstructed = model(X_test)
    reconstruction_loss = criterion(X_test_reconstructed, X_test)
    print(f'Test Reconstruction Loss: {reconstruction_loss.item():.4f}')
