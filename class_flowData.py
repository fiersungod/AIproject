class flowData:
    def __init__(self,s:str):
        s = s.split(",")
        self.timestamp = float(s[0])
        self.source_IP = s[1]
        self.destination_IP = s[2]
        self.source_Port = int(s[3])
        self.destination_Port = int(s[4])
        self.protocol = int(s[5])
        self.duration = float(s[6])
        self.source_bytes = int(s[7])
        self.destination_bytes = int(s[8])
        self.source_efficiency = float(s[9])
        self.destination_efficiency = float(s[10])
        self.source_packets = int(s[11])
        self.destination_packets = int(s[12])
        self.source_seq = int(s[13])
        self.destination_seq = int(s[14])
        if len(s) >= 16: 
            self.answer = int(s[15])
            if len(s) >= 17: self.attack_category = s[16]
            else: self.attack_category = ""
        else:
            self.answer = 0
            self.attack_category = ""

    def to_list(self,last_time=0):
        result = [(self.timestamp - last_time) / 1000]
        #result += [int(i)/255 for i in self.source_IP.split(".")]
        #result += [int(i)/255 for i in self.destination_IP.split(".")]
        #result.append(self.source_Port/65535)
        #result.append(self.destination_Port/65535)
        result.append(self.protocol)
        result.append(self.duration)
        result.append(self.source_bytes/1e6)
        result.append(self.destination_bytes/1e6)
        result.append(self.source_efficiency/1e6)
        result.append(self.destination_efficiency/1e6)
        result.append(self.source_packets/100)
        result.append(self.destination_packets/100)
        result.append(self.source_seq/1e10)   
        result.append(self.destination_seq/1e10)
        return result