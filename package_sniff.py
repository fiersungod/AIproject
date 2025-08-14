import datetime
from scapy.all import UDP,TCP, IP, sniff, get_if_list

#-----------Settings----------------
#csv file will store at this path
recordPath = r'D:\Code\project'
#Time length that program will sniff
timeOut = 10

#Network interface will program sniff.
#If program can't sniff any data, try get_if_list() and modify iFace like down below:
#iFace = r"\Device\NPF_{4273AC39-B6B8-469C-9CAD-751D786668DC}"
iFace = get_if_list()
#-----------------------------------

def packet_callback(packet):
    if (packet.haslayer(UDP)) and packet.haslayer(IP):
        packageData = [str((datetime.datetime.now()-now).total_seconds()),str(packet[IP].src),str(packet[IP].dst),str(packet[UDP].sport),str(packet[UDP].dport),str(packet[IP].len)]
        dataTemp.append(packageData)
    elif (packet.haslayer(TCP)) and packet.haslayer(IP):
        packageData = [str((datetime.datetime.now()-now).total_seconds()),str(packet[IP].src),str(packet[IP].dst),str(packet[TCP].sport),str(packet[TCP].dport),str(packet[IP].len),str(len(str(packet[TCP].flags))),str(packet[TCP].seq)]
        dataTemp.append(packageData)
        
#UDP ['time','source_IP','destination_IP','source_Port','destination_Port','package_length']
#TCP ['time','source_IP','destination_IP','source_Port','destination_Port','package_length','flag_count','sequence_number']
dataTemp = []
now = datetime.datetime.now()
sniff(iface=iFace,filter="udp or tcp", prn=packet_callback, store=0,timeout=timeOut)
recordPath += now.strftime("\\%Y%m%d%H%M%S.csv")
with open(recordPath,'w',encoding='utf-8') as file:
    for data in dataTemp:
        file.writelines(','.join(data) + '\n')
