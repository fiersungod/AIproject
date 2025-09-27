import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
import GAT_VAE
import flow_sniff
from class_flowData import flowData
from class_gatData import build_graph_from_flow

stop_event = threading.Event()

def realtime_detect(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model,optimizer = GAT_VAE.initial_model(device=device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("realtime_detction: Checkpoint not found. Please check model_path.")
        raise FileNotFoundError

    # Open a thead to sniff flows and start predicting at the same time
    sniff_thread = threading.Thread(target=flow_sniff.sniff_flow, daemon=True)
    sniff_thread.start()
    with torch.no_grad():
        model.eval()

        window_size = GAT_VAE.WINDOW_SIZE
        time_threshold = GAT_VAE.TIME_THRESHOLD
        loss_threshold = GAT_VAE.LOSS_THRESHOLD

        # Initial with normal flow
        start_flag = True
        while start_flag and not stop_event.is_set():
            while len(flow_sniff.completed_flows) < window_size:
                time.sleep(1)
            current_flow = flow_sniff.drop_flows(window_size)
            if current_flow is None or len(current_flow) < window_size:
                time.sleep(1)
                continue
            pyg_data = [flowData(i) for i in current_flow]
            pyg_data = build_graph_from_flow(pyg_data, time_threshold=time_threshold).to(device)
            recon_x, mu, logvar, gat_out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            BCEloss, KLloss = GAT_VAE.vae_loss(recon_x, gat_out, mu, logvar)
            loss = BCEloss + KLloss
            if loss.item() < loss_threshold:
                start_flag = False
                print("Initial normal flow established.")
        
        # Start predicting
        while not stop_event.is_set():
            if len(flow_sniff.completed_flows) == 0:
                time.sleep(1)
            input_flow = flow_sniff.drop_flows(1)
            if input_flow is None:
                time.sleep(1)
                continue
            test_flow = current_flow[1:]
            test_flow.append(input_flow[0])
            pyg_data = [flowData(i) for i in test_flow]
            pyg_data = build_graph_from_flow(pyg_data, time_threshold=time_threshold).to(device)
            recon_x, mu, logvar, gat_out = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            BCEloss, KLloss = GAT_VAE.vae_loss(recon_x, gat_out, mu, logvar)
            loss = BCEloss + KLloss
            if loss.item() > loss_threshold:
                print(f"Anomaly detected! Loss: {loss.item()}")
            else:
                print(f"Normal flow. Loss: {loss.item()}")
                current_flow = test_flow

    sniff_thread.join()
    print("Stopped realtime detection.")

def stop_realtime_detection():
    stop_event.set()
    
if __name__ == "__main__":
    model_path = "save_model\\gat_vae_model.pth"
    realtime_detect(model_path)





