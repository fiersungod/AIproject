import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import class_gatData as g
import torch
from sklearn.metrics import classification_report
import GAT_VAE as gv
import random
from NB15_to_flow import get_flow_data
from class_gatData import build_graph_from_flow
from GAT_VAE import initial_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model,optimizer = initial_model(device=device)

checkpoint_path = r"C:\Users\austi\OneDrive\Desktop\project - flow_base_test\gat_vae_model.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded.")
except FileNotFoundError:
    print("Checkpoint not found. Please check checkpoint_path.")
    raise FileNotFoundError

test_paths = r"C:\Users\austi\OneDrive\Desktop\專題-test\UNSW-NB15_2.csv"

#use nb15 for now
flow_data = get_flow_data(training=False,file_path=test_paths)
flow_data = [flow_data[i:i+15] for i in range(0, len(flow_data), 15)]
answers = []
pyg_data = []

for i in flow_data:
    pyg_data.append(build_graph_from_flow(i,time_threshold=1).to(device))
    ans = 0
    for j in i:
        if j.answer == 1:
            ans = 1
            break
    answers.append(ans)
model.eval()
with torch.no_grad():
    predicts = []
    total_loss = []
    nor,att = 0,0
    threshold = 2.5
    for i in range(len(pyg_data)):
        recon_x, mu, logvar,gat_out = model(pyg_data[i].x, pyg_data[i].edge_index,pyg_data[i].edge_attr)
        BCEloss, KLloss = gv.vae_loss(recon_x, gat_out, mu, logvar)
        loss = BCEloss + KLloss
        #loss = gv.vae_loss(recon_x, gat_out, mu, logvar)
        #print("BCELoss:", BCEloss, "KLloss:", KLloss, "total loss:",loss)
        total_loss.append(loss.item())
        print(loss.item(),end=",")
        if loss.item() > threshold:
            predicts.append(1)
        elif loss.item() <= threshold:
            predicts.append(0)
    print("")
    print(classification_report(answers, predicts))

    total_loss = pd.DataFrame(total_loss)
    scored = pd.DataFrame()
    scored["total_loss"] = np.abs(total_loss)
    
    plt.figure()
    sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
    plt.show()