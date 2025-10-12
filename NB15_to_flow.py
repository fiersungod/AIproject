import class_flowData

def get_flow_data(training=False,file_path=""):
    if file_path == "":
        raise FileNotFoundError("get_flow_data : Please provide a valid file_path.")
    skip_count = 0
    dataStrList = []
    with open(file_path,"r",encoding='utf_8_sig') as f:
        for line in f:
            if training and skip_count >= 1:
                skip_count -= 1
                dataStrList.pop()
                continue
            dataStr = ""
            csvData = line.split(",")
            csvData[48] = csvData[48].strip()
            if (csvData[4] in ["tcp","udp"]) and ((not training) or (csvData[48] == '0')):
                dataStr += f"{csvData[28]},{csvData[0]},{csvData[2]},{csvData[1]},{csvData[3]}"
                if csvData[4] == "udp": dataStr += ",0"
                elif csvData[4] == "tcp": dataStr += ",1"
                dataStr += f",{csvData[6]},{csvData[7]},{csvData[8]},{csvData[14]},{csvData[15]},{csvData[16]},{csvData[17]},{csvData[20]},{csvData[21]},{csvData[48]}"
                if csvData[48] != '0':
                    dataStr += f",{csvData[47]}"
                dataStrList.append(dataStr)
            if training and csvData[48] == '1':
                skip_count += 5
        f.close()
    flow_data = [class_flowData.flowData(i) for i in dataStrList]
    return flow_data
