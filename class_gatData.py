import torch
from torch_geometric.data import Data
from class_flowData import flowData

def build_graph_from_flow(packets: list[flowData],time_threshold=1,device='cpu'):
    #x
    start = packets[0].timestamp
    x = [p.to_list(start) for p in packets]

    #edge
    N = len(packets)
    edge_index = []
    edge_attr = []
    for i in range(N):
        for j in range(i+1, N):
            time_diff = abs(packets[j].timestamp - packets[i].timestamp)
            if time_diff < time_threshold:
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

    if edge_index == []:
        print("Constructing graph : No edges found, consider increasing the time threshold.")
        for i in range(N):
            edge_index.append([i, i])
            edge_attr.append([1, 0])
    # 轉換為 tensor 格式
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape: (2, E)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # shape: (E, D)
    
    # 建立 PyG 的 Data 物件
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    return data

def load_csv_data(csv_path,max_size=1000000):
    with open(csv_path) as f:
        next(f)
        data = []
        counter = 0
        for line in f:
            data.append(flowData(line))
            counter += 1
            if counter >= max_size:
                break

    return data

if __name__ == "__main__":
    path = "local_data_set//flow_20250830211141.csv"
    flow_data = load_csv_data(path)
    for i in flow_data:
        pyg_data = build_graph_from_flow(i,time_threshold=1)
    print("success!")