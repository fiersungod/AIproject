import torch
#單一個UDP封包資料
class udpData:
    def __init__(self):
        self.time = 0
        self.source_IP = ""
        self.destination_IP = ""
        self.source_Port = 0
        self.destination_Port = 0
        self.package_length = 0

    def __init__(self,s:str):
        s = s.split(",")
        self.time = float(s[0])
        self.source_IP = s[1]
        self.destination_IP = s[2]
        self.source_Port = int(s[3])
        self.destination_Port = int(s[4])
        self.package_length = int(s[5])

    def to_list(self,last_time=0):
        result = [int((self.time - last_time) * 1000000)]
        result += [int(i) for i in self.source_IP.split(".")]
        result += [int(i) for i in self.destination_IP.split(".")]
        result.append(self.source_Port)
        result.append(self.destination_Port)
        result.append(self.package_length)
        return result
    
def csv_to_tensor(csv_path,package_size = 5,step = 2):
    with open(csv_path) as f:
        data,result = [],[]
        now = 0
        for i in f:
            data.append(udpData(i))
        for i in range(len(data)):
            try:
                temp = []
                for i in range(package_size):
                    if ((now + i -1) > 0):
                        last_time = data[now+i-1].time
                    else:
                        last_time = data[now+i].time
                    temp += data[now + i].to_list(last_time)
                result.append(temp)
                now += step
            except:
                break
    if result == []:
        raise "csv file NOT found or csv headers should be removed!" 
    return torch.tensor(result)

if __name__ == "__main__":
    path = r"C:\Users\austi\OneDrive\Desktop\AIproject\20250417140400-39.csv"
    print("Testing csv_to_tensor...")
    print("Print tensor data if test is success.")
    try:
        T = csv_to_tensor(path,1,1)
        print(f"Generated {len(T)} datas.")
        print(T)
    except FileNotFoundError:
        raise Exception("csv file NOT found!")
    except ValueError:
        raise Exception("csv headers SHOULD BE removed!")
