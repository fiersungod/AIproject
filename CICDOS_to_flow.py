from datetime import datetime
import class_flowData

def get_flow_data(file_path=""):
    if file_path == "":
        raise FileNotFoundError("get_flow_data : Please provide a valid file_path.")
    count = 0
    dataStrList = []
    timeStart = datetime.strptime("12:36:57.628025", "%H:%M:%S.%f")
    with open(file_path,"r",encoding='utf_8_sig') as f:
        next(f)
        for line in f:
            count += 1
            try:
                dataStr = ""
                csvData = line.split(",")
                timeStamp = datetime.strptime(csvData[7][11:], "%H:%M:%S.%f") - timeStart
                dataStr += (format(timeStamp.total_seconds(),'.6f').rstrip('0'))
                dataStr += f",{csvData[2]},{csvData[4]},{csvData[3]},{csvData[5]},0"
                dataStr += f",{str(int(float(csvData[8])//1000))},{str(int(float(csvData[11])))},{str(int(float(csvData[12])))},{str(float(csvData[21])*8)},{str(float(csvData[21])*8)},{csvData[9]},{csvData[10]},0,0,1"
                dataStrList.append(dataStr)
            except:
                #print(f"Error processing line {count}")
                continue
        f.close()
    flow_data = [class_flowData.flowData(i) for i in dataStrList]
    return flow_data

if __name__ == "__main__":
    r_file_path = r'C:\Users\austi\OneDrive\Desktop\專題-test\DrDoS_UDP.csv'
    flow_data = get_flow_data(file_path=r_file_path)

