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

checkpoint_path = r'project\save_model\gat_vae_model.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded.")
except FileNotFoundError:
    print("Checkpoint not found. Starting training from scratch.")
    checkpoint_path = r'project\gat_vae_model.pth'


test_paths = [r"project\local_data_set\20250515230558-40.csv",
              r"project\local_data_set\20250515230632-40.csv",
              r"project\local_data_set\20250515230706-40.csv",
              r"project\local_data_set\20250515230740-40.csv"]


test_paths = [r'C:\Users\austi\OneDrive\Desktop\專題-test\CDC_.csv']

#test_paths = [r"project\local_data_set\20250515230817-40.csv"]

#test_paths = [r"project\local_data_set\20250502160612-39.csv"]

udp_datas = []
for i in test_paths:
    udp_datas += g.load_csv_data(i,50)
answers = []
pyg_data = []
for i in udp_datas:
    pyg_data.append(g.build_graph_from_packets(i).to(device))
    ans = 0
    for j in i:
        if j.answer == 1:
            ans = 1
            break
    answers.append(ans)
model.eval()
with torch.no_grad():
    total_loss = []
    nor,att = 0,0
    tp,fp,tn,fn=0,0,0,0
    threshold = 26000
    #random.shuffle(pyg_data)
    for i in range(len(pyg_data)):
        recon_x, mu, logvar,gat_out = model(pyg_data[i].x, pyg_data[i].edge_index,pyg_data[i].edge_attr)
        """
        print(recon_x)
        print("===")
        print(mu)
        print("===")
        print(logvar,)
        print("===")
        print(gat_out)
        """
        BCEloss, KLloss = gv.vae_loss(recon_x, gat_out, mu, logvar)
        loss = BCEloss + KLloss
        #loss = gv.vae_loss(recon_x, gat_out, mu, logvar)
        print("BCELoss:", BCEloss, "KLloss:", KLloss, "total loss:",loss)
        total_loss.append(loss.item())
        if answers[i] == 1: 
            att += 1
            if loss.item() > threshold:
                tp += 1
            else:
                fn += 1
        else:
            nor += 1
            if loss.item() > threshold:
                fp += 1
            else:
                tn += 1
    total_loss = pd.DataFrame(total_loss)
    scored = pd.DataFrame()
    scored["total_loss"] = np.abs(total_loss)
    
    print("normal sample: " , nor)
    print("attack sample: " , att)

    print("tp: ",tp)
    print("tn: ",tn)
    print("fp: ",fp)
    print("fn: ",fn)
    print("acc: ",(tp+tn)/(tp+tn+fp+fn))
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    print("pre: ",pre)
    print("rec: ",rec)
    print("f1-score: ",(2*pre*rec)/(pre+rec))

    plt.figure()
    sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
    plt.show()