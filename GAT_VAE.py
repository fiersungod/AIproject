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
from NB15_to_flow import get_flow_data
from class_gatData import build_graph_from_flow
from sklearn.metrics import classification_report

# ---- GAT 模型（圖神經網絡） ----
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):# 更改頭數(2、4、8、16)調整為最佳狀態
        super(GATModel, self).__init__()
        # 1st GAT layer
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        # 2nd GAT layer
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
        self.fce1 = nn.Linear(11, 64)
        self.fce2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, z_dim)
        self.fc_logvar = nn.Linear(32, z_dim)

        # Decoder
        self.fcd3 = nn.Linear(z_dim, 32)
        self.fcd2 = nn.Linear(32, 64)  
        self.fcd1 = nn.Linear(64, 11)

    def encode(self, x):
        h = F.relu(self.fce1(x))  # Encoding layer
        h = F.relu(self.fce2(h)) 
        mu = self.fc_mu(h)  # Mean of latent space
        logvar = self.fc_logvar(h)  # Log variance of latent space
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fcd3(z))  # Decoder layer
        h = F.relu(self.fcd2(h))
        return self.fcd1(h)

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
def train(model, data, optimizer, epoch=100,early_stopping_thereshold=1):
   pocket = [i for i in range(len(data))]
   random.shuffle(pocket)
   bin = []  #暫存訓練中已使用過的資料
   datas = {i : v for i, v in enumerate(data)} 
   stack = [] # 用於early stopping的堆疊
   stopFlag = False
   model.train()
   for e in range(epoch):
        if (pocket == []):
            #if stopFlag:
                #break
            pocket = bin
            bin = []
            random.shuffle(pocket)
        num = pocket.pop()
        bin.append(num)
        optimizer.zero_grad()
        recon_x, mu, logvar,gat_out = model(datas[num].x, datas[num].edge_index,datas[num].edge_attr)
        BCEloss, KLloss = vae_loss(recon_x, gat_out, mu, logvar)
        loss = BCEloss + KLloss
        loss.backward()
        optimizer.step()
        #Early Stopping
        if loss <= early_stopping_thereshold and not stopFlag:
            stack.append(e)
            if len(stack) >= 100:
                arr = [stack[i] - stack[i - 1] for i in range(1, len(stack))]
                if all(x == 1 for x in arr):
                    #stopFlag = True
                    print(f"Early stopping at epoch {e} with loss {loss.item()}")
                    break
                else:
                    stack.pop(0)
        if e % 100 == 0:
            print(f"Epoch {e}/{epoch}, BCELoss: {BCEloss.item()}, KL: {KLloss.item()}, Loss: {loss.item()}")

def initial_model(device):
    model = GAT_VAE(in_channels= 11, gat_hidden=32, gat_out=11, z_dim=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
    return model,optimizer


WINDOW_SIZE = 5
TIME_THRESHOLD = 60
LOSS_THRESHOLD = 5.0
def trainModel(paths=None,model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if paths is None or paths == "" or paths == []:
        raise FileNotFoundError("GAT_VAE: trainModel : File_path is invaild.")

    flow_data = []
    pyg_data = []
    for file_path in paths:
        if "NB15" in file_path:
            data = get_flow_data(training=True,file_path=file_path)
            flow_data += data
        else:
            data = g.load_csv_data(file_path)
            flow_data += data
    flow_data = [flow_data[i:i+WINDOW_SIZE] for i in range(0, len(flow_data), WINDOW_SIZE)]
    for i in flow_data:
        pyg_data.append(build_graph_from_flow(i,time_threshold=TIME_THRESHOLD).to(device))

    model,optimizer = initial_model(device=device)
    if model_path is not None:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Checkpoint loaded.")
        except FileNotFoundError:
            print("GAT_VAE: Checkpoint not found. Please check checkpoint_path.")
            raise FileNotFoundError

    # 開始訓練
    epochs = 5*len(pyg_data)
    train(model, pyg_data, optimizer,epoch=epochs,early_stopping_thereshold=LOSS_THRESHOLD)

    # 測試、保存模型
    checkpoint_path = "save_model\\gat_vae_model.pth"
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print("Final model saved.")

    # 測試模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("start testing")
    test_path = r"C:\Users\austi\OneDrive\Desktop\專題-test\UNSW-NB15_2.csv"

    try:
        flow_data = get_flow_data(training=False,file_path=test_path)
    except FileNotFoundError:
        print("Didn't find test data, skip testing.")
    else:
        flow_data = [flow_data[i:i+WINDOW_SIZE] for i in range(0, len(flow_data), WINDOW_SIZE)]
        pyg_data = []
        answers = []
        predicts = []
        for i in flow_data:
            if any(j.answer == 1 for j in i):
                answers.append(1)
            else:
                answers.append(0)
            pyg_data.append(g.build_graph_from_flow(i,time_threshold=TIME_THRESHOLD).to(device))
        model.eval()
        with torch.no_grad():
            total_loss = []
            for data in pyg_data:
                recon_x, mu, logvar,gat_out = model(data.x, data.edge_index,data.edge_attr)
                BCEloss, KLloss = vae_loss(recon_x, gat_out, mu, logvar)
                loss = BCEloss + KLloss
                total_loss.append(loss.item())
                if loss.item() > LOSS_THRESHOLD:
                    predicts.append(1)
                else:
                    predicts.append(0)
            total_loss = pd.DataFrame(total_loss)
            scored = pd.DataFrame()
            scored["total_loss"] = np.abs(total_loss)

            print(classification_report(answers, predicts))

            # 繪製分佈圖
            plt.figure()
            sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
            plt.show()

if __name__ == "__main__":
    trainModel()