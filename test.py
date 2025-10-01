import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import class_gatData as g
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import GAT_VAE as gv
from class_gatData import build_graph_from_flow
from GAT_VAE import initial_model
import NB15_to_flow
import CICDOS_to_flow

def test(test_paths=None):
    WINDOW_SIZE = gv.WINDOW_SIZE
    TIME_THRESHOLD = gv.TIME_THRESHOLD
    LOSS_THRESHOLD = gv.LOSS_THRESHOLD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model,optimizer = initial_model(device=device)

    checkpoint_path = "save_model\\gat_vae_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found. Please check checkpoint_path.")
        raise FileNotFoundError

    if test_paths is None:
        raise FileNotFoundError("Please check test_paths.")

    model.eval()
    with torch.no_grad():
        if "NB15" in test_paths:
            flow_data = NB15_to_flow.get_flow_data(training=False,file_path=test_paths)
        else:
            flow_data = CICDOS_to_flow.get_flow_data(file_path=test_paths)
        data_count = (len(flow_data) - 1) // WINDOW_SIZE
        print(f"Total data count: {data_count}")
        trigger = int(data_count * 0.1)   
        answers = []
        predicts = []
        total_loss = []
        now = 0
        while data_count > 0:
            current_flow  = flow_data[now:now + WINDOW_SIZE]
            now += WINDOW_SIZE
            answers.append(1 if any(f.answer == 1 for f in current_flow) else 0)
            pyg_data = build_graph_from_flow(current_flow,time_threshold=TIME_THRESHOLD).to(device)
            recon_x, mu, logvar,gat_out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            BCEloss, KLloss = gv.vae_loss(recon_x, gat_out, mu, logvar)
            loss = BCEloss + KLloss
            if loss.item() > LOSS_THRESHOLD:
                predicts.append(1)
            else:
                predicts.append(0)
            total_loss.append(loss.item())
            data_count -= 1
            if data_count % trigger == 0:
                print(f"Remaining test data: {data_count}")
        print("Test with non-overlapping window approach (old).")

        #print results
        print(classification_report(answers, predicts))
        total_loss = pd.DataFrame(total_loss)
        scored = pd.DataFrame()
        scored["total_loss"] = np.abs(total_loss)
        plt.figure()
        sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
        plt.show()

        fpr, tpr, thresholds = roc_curve(answers, total_loss)
        plt.plot(thresholds, tpr, label="TPR")
        plt.plot(thresholds, 1-fpr, label="1-FPR")
        plt.xlabel("Threshold")
        plt.legend()
        plt.show()

def test_windows(test_paths=None):
    WINDOW_SIZE = gv.WINDOW_SIZE
    TIME_THRESHOLD = gv.TIME_THRESHOLD
    LOSS_THRESHOLD = gv.LOSS_THRESHOLD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model,optimizer = initial_model(device=device)

    checkpoint_path = "save_model\\gat_vae_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("Checkpoint not found. Please check checkpoint_path.")
        raise FileNotFoundError

    if test_paths is None:
        raise FileNotFoundError("Please check test_paths.")

    model.eval()
    with torch.no_grad():
        #initial with normal flow
        if "NB15" in test_paths:
            flow_data = NB15_to_flow.get_flow_data(training=False,file_path=test_paths)
        else:
            flow_data = CICDOS_to_flow.get_flow_data(file_path=test_paths)
        data_count = len(flow_data) - 1
        print(f"Total data count: {data_count}")
        trigger = int(data_count * 0.1)   
        start_flag = True
        while start_flag:
            current_flow  = flow_data[:WINDOW_SIZE]
            flow_data[:] = flow_data[WINDOW_SIZE:]
            data_count -= WINDOW_SIZE
            pyg_data = build_graph_from_flow(current_flow,time_threshold=TIME_THRESHOLD).to(device)
            recon_x, mu, logvar,gat_out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            BCEloss, KLloss = gv.vae_loss(recon_x, gat_out, mu, logvar)
            loss = BCEloss + KLloss
            if loss.item() < LOSS_THRESHOLD:
                start_flag = False

        #start predicting
        answers = []
        predicts = []
        total_loss = []
        while data_count > 0:
            test_flow = current_flow[1:]
            test_data = flow_data.pop(0)
            answers.append(1 if test_data.answer == 1 else 0)
            test_flow.append(test_data)
            pyg_data = build_graph_from_flow(test_flow,time_threshold=TIME_THRESHOLD).to(device)
            recon_x, mu, logvar,gat_out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            BCEloss, KLloss = gv.vae_loss(recon_x, gat_out, mu, logvar)
            loss = BCEloss + KLloss
            if loss.item() > LOSS_THRESHOLD:
                predicts.append(1)
            else:
                predicts.append(0)
                current_flow = test_flow.copy()
            total_loss.append(loss.item())
            data_count -= 1
            if data_count % trigger == 0:
                print(f"Remaining test data: {data_count}")
            #print(loss.item(),end=",")

        print("Testing with sliding window approach (new).")

        #print results
        print(classification_report(answers, predicts))
        total_loss = pd.DataFrame(total_loss)
        scored = pd.DataFrame()
        scored["total_loss"] = np.abs(total_loss)
        plt.figure()
        sns.histplot(scored["total_loss"], bins=10, kde=True, color='blue')  # 使用 seaborn 繪製分佈圖
        plt.show()

        fpr, tpr, thresholds = roc_curve(answers, total_loss)
        plt.plot(thresholds, tpr, label="TPR")
        plt.plot(thresholds, 1-fpr, label="1-FPR")
        plt.xlabel("Threshold")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    test_paths = r"C:\Users\austi\OneDrive\Desktop\專題-test\UNSW-NB15_2.csv"
    test_paths = r"C:\Users\austi\OneDrive\Desktop\專題-test\DrDoS_UDP.csv"
    test(test_paths=test_paths)
    #test_windows(test_paths=test_paths)