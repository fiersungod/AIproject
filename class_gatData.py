import class_udpData as u
import torch
from torch_geometric.data import Data

def build_graph_from_packets(packets: list[u.udpData],time_threshold=1,device='cpu'):
    """
    packet_features: (N, 12) tensor
    timestamps: (N,) tensor
    time_threshold: float (in seconds)
    """
    #x
    x = [p.to_list() for p in packets]

    #edge
    N = len(packets)
    edge_index = []
    edge_attr = []
    for i in range(N):
        for j in range(i+1, N):
            time_diff = abs(packets[j].time - packets[i].time)
            if time_diff <= time_threshold:
                # 建立雙向邊 (i -> j) and (j -> i)
                edge_index.append([i, j])
                edge_index.append([j, i])

                # 設定 edge_attr，時間越近數值越接近1
                attr = []
                attr.append(1 - time_diff/time_threshold)

                # 設定 edge_attr，互相傳輸為2，相同來源為1，無關係為0
                if (packets[i].source_IP == packets[j].destination_IP and packets[i].destination_IP == packets[j].source_IP):
                    attr.append(2)
                elif (packets[i].source_IP == packets[i].destination_IP and packets[j].destination_IP == packets[j].source_IP):
                    attr.append(1)
                else:
                    attr.append(0)

                edge_attr.append(attr)
                edge_attr.append(attr)
            else:
                break

    # 轉換為 tensor 格式
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape: (2, E)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # shape: (E, D)
    
    # 建立 PyG 的 Data 物件
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    return data

def load_csv_data(csv_path,size=200):
    with open(csv_path) as f:
        data,temp = [],[]
        counter = 0
        for i in f:
            temp.append(u.udpData(i))
            counter += 1
            if counter == size:
                data.append(temp)
                temp = []
                counter = 0
        #if temp != []:
        #    data.append(temp)
    return data

if __name__ == "__main__":
    path = r"C:\Users\austi\OneDrive\Desktop\AIproject\20250417140400-39.csv"
    udp_data = load_csv_data(path)
    pyg_data = build_graph_from_packets(udp_data,time_threshold=0.3)
    print("success!")