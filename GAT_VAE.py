import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import class_gatData as g

# ---- GAT 模型（圖神經網絡） ----
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=2):# 更改頭數(2、4、8、16)調整為最佳狀態
        super(GATModel, self).__init__()
        # 1st GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        # 2nd GAT layer
        # 單頭注意力，將前一層所有頭的輸出拼接起來
        # 方便後續處理
        # hidden_channels * num_heads = out_channels
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1)

    def forward(self, x, edge_index,edge_attr):
        x = F.relu(self.gat1(x, edge_index,edge_attr)) # Apply first GAT layer
        x = self.gat2(x, edge_index,edge_attr)  # Apply second GAT layer
        return x

# ---- VAE 模型（變分自編碼器） ----
class VAE(nn.Module):
    def __init__(self, z_dim=16):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        
        # Encoder (Latent space parameters: mu and logvar)
        # 擴增特徵維度，方便學習更多重點特徵
        self.fc1 = nn.Linear(64, 128)  # Input size from GAT (64 features)
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logvar = nn.Linear(128, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, 128)
        self.fc4 = nn.Linear(128, 64)

    def encode(self, x):
        h = F.relu(self.fc1(x))  # Encoding layer
        mu = self.fc_mu(h)  # Mean of latent space
        logvar = self.fc_logvar(h)  # Log variance of latent space
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))  # Decoder layer
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ---- VAE Loss Function (KL + MSE) ----
def vae_loss(recon_x, x, mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss (MSE)
    # KL divergence loss
    # Standard normal distribution: N(0, I)
    # KL divergence (D_KL(q(z|x)||p(z)))
    # This term encourages z to follow a normal distribution
    # where mu=0 and logvar=0
    # see https://arxiv.org/abs/1312.6114
    # logvar is the log of the variance
    # mu is the mean of the latent variable
    # For simplicity, we assume the variance is 1
    # Kullback-Leibler divergence term
    # KL divergence between normal and learned latent variable distribution
    # (this will make the latent distribution similar to a normal one)
    # use the following formula:
    # KL(q(z|x)||p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # This encourages the posterior distribution q(z|x) to be close to N(0, I)
    # which is an isotropic Gaussian distribution.
    # Note: `logvar` is the logarithm of the variance.
    # Reference: Kingma & Welling (2013)
    # https://arxiv.org/pdf/1312.6114.pdf
    # https://stackoverflow.com/questions/42902906/understanding-kl-divergence-in-vae
    # In PyTorch, `logvar` is the log of the variance.
    # So we can use the following formula for KL divergence:
    # KL(q(z|x)||p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Reference:
    # https://en.wikipedia.org/wiki/Variational_autoencoder#Loss_function
    # The result will be summed over the batch
    # Negative log likelihood:
    # We try to minimize this function.
    # It's similar to the conventional log likelihood, but with an additional KL term.
    # L(x, z) = L_vae(x, z) + L_kl(x, z)
    # Return this loss value
    # Reconstruction term + KL divergence term
    # sum across the batch for each data point
    # Add these two terms and return the final loss
    return BCE, -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# ---- 整體模型整合（GAT + VAE） ----
class GAT_VAE(nn.Module):
    def __init__(self, in_channels, gat_hidden, gat_out, z_dim=16):
        super(GAT_VAE, self).__init__()
        self.gat = GATModel(in_channels, gat_hidden, gat_out)
        self.vae = VAE(z_dim)

    def forward(self, x, edge_index,edge_attr):
        # GAT 層提取節點嵌入
        gat_out = self.gat(x, edge_index,edge_attr)
        # VAE 層進行重建
        recon_x, mu, logvar = self.vae(gat_out)
        return recon_x, mu, logvar,gat_out

# ---- 訓練流程 ----
def train(model, data, optimizer, epoch=100):
   pocket = [i for i in range(len(data))]
   random.shuffle(pocket)
   bin = []
   datas = {i : v for i, v in enumerate(data)}
   model.train()
   for e in range(epoch):
        if (pocket == []):
            pocket = bin
            bin = []
            random.shuffle(pocket)
        num = pocket.pop()
        bin.append(num)
        optimizer.zero_grad()
        recon_x, mu, logvar,gat_out = model(datas[num].x, datas[num].edge_index,datas[num].edge_attr)
        #recon_x, mu, logvar,gat_out = model(i.x, i.edge_index,i.edge_attr)
        BCEloss, KLloss = vae_loss(recon_x, gat_out, mu, logvar)
        loss = BCEloss + KLloss
        #loss = F.mse_loss(recon_x, i.x, reduction='sum') + KL
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print(f"Epoch {e}/{epoch}, BCELoss: {BCEloss.item()}, KL: {KLloss.item()}, Loss: {loss.item()}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = [r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230558-40.csv",
             r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230632-40.csv",
             r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230706-40.csv",
             r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230740-40.csv"]
    
    udp_datas = []
    for i in paths:
        udp_datas += g.load_csv_data(i,50)
    pyg_data = []
    for i in udp_datas:
        pyg_data.append(g.build_graph_from_packets(i,time_threshold=1).to(device))
    print(pyg_data)

    model = GAT_VAE(in_channels= 12, gat_hidden=32, gat_out=64, z_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 開始訓練
    epochs = 50*len(pyg_data)
    #epochs = 50
    train(model, pyg_data, optimizer,epoch=epochs)

    # 測試、保存模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_paths = [r"C:\Users\austi\OneDrive\Desktop\AIproject\local_data_set\20250417140400-39.csv"]

    udp_datas = []
    for i in test_paths:
        udp_datas += g.load_csv_data(i,50)
    pyg_data = []
    for i in udp_datas:
        pyg_data.append(g.build_graph_from_packets(i,time_threshold=1).to(device))
    model.eval()
    with torch.no_grad():
        total_loss = []
        for data in pyg_data:
            recon_x, mu, logvar,gat_out = model(data.x, data.edge_index,data.edge_attr)
            """
            print(recon_x)
            print("===")
            print(mu)
            print("===")
            print(logvar,)
            print("===")
            print(gat_out)
            """
            loss = vae_loss(recon_x, gat_out, mu, logvar)
            print("total loss:",loss)
            total_loss.append(loss.item())
        total_loss = pd.DataFrame(total_loss)
        scored = pd.DataFrame()
        scored["total_loss"] = np.abs(total_loss)

        plt.figure()
        sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
        plt.show()

        checkpoint_path = r"C:\Users\austi\OneDrive\Desktop\AIP_test2\gat_vae_model.pth"
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print("Final model saved.")
