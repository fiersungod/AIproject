import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dummy dataset (replace with real network packet data)
def generate_dummy_data(num_samples=10000, num_features=20):
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)  # 0 for normal, 1 for abnormal
    return X, y

# Define the AutoEncoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load dataset
X, y = generate_dummy_data()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Initialize model, loss function, and optimizer
model = AutoEncoder(input_size=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate model
with torch.no_grad():
    X_test_reconstructed = model(X_test_tensor)
    reconstruction_loss = criterion(X_test_reconstructed, X_test_tensor)
    print(f'Test Reconstruction Loss: {reconstruction_loss.item():.4f}')
