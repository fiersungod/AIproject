import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import class_gatData as g
import torch
import GAT_VAE as gv
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = gv.GAT_VAE(in_channels= 12, gat_hidden=32, gat_out=64, z_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint_path = r'C:\Users\austi\OneDrive\Desktop\AIP_test2\gat_vae_model.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded.")
except FileNotFoundError:
    print("Checkpoint not found. Starting training from scratch.")
    checkpoint_path = r'C:\Users\austi\OneDrive\Desktop\AIP_test2\gat_vae_model.pth'


test_paths = [r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230558-40.csv",
              r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230632-40.csv",
              r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230706-40.csv",
              r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230740-40.csv"]


#test_paths = [r'C:\Users\austi\OneDrive\Desktop\專題-test\CDC_.csv']

test_paths = [r"C:\Users\austi\OneDrive\Desktop\AIP_test2\20250515230817-40.csv"]

test_paths = [r"C:\Users\austi\OneDrive\Desktop\專題-test\20250502160612-39.csv"]

udp_datas = []
for i in test_paths:
    udp_datas += g.load_csv_data(i,50)
pyg_data = []
for i in udp_datas:
    pyg_data.append(g.build_graph_from_packets(i).to(device))
model.eval()
with torch.no_grad():
    total_loss = []
    #random.shuffle(pyg_data)
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
        loss = gv.vae_loss(recon_x, gat_out, mu, logvar)
        print("total loss:",loss)
        total_loss.append(loss.item())
    total_loss = pd.DataFrame(total_loss)
    scored = pd.DataFrame()
    scored["total_loss"] = np.abs(total_loss)
    
    plt.figure()
    sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
    plt.show()