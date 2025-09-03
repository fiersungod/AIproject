class tcpData:
    def __init__(self,s:str):
        s = s.split(",")
        self.time = float(s[0])
        self.source_IP = s[1]
        self.destination_IP = s[2]
        self.source_Port = int(s[3])
        self.destination_Port = int(s[4])
        self.package_length = int(s[5])
        self.flag_count = int(s[6])
        self.sequence_number = int(s[7])
        if len(s) >= 9: self.answer = int(s[8])
        else: self.answer = 0

    def to_list(self,last_time=0):
        result = [(self.time - last_time) * 1000]
        result += [int(i)/100 for i in self.source_IP.split(".")]
        result += [int(i)/100 for i in self.destination_IP.split(".")]
        result.append(self.source_Port/10000)
        result.append(self.destination_Port/10000)
        result.append(self.package_length/1000)
        result.append(self.flag_count)
        result.append(self.sequence_number/10000000000)
        return result