from scapy.all import sniff,get_if_list, IP, TCP, UDP
from collections import defaultdict
import pandas as pd
import time
import datetime
import threading

FLOW_TIMEOUT = 15   # Flow timeout in seconds
RECORD_TIME = 3660  # Total sniffing time in seconds

class flowKey:
    def __init__(self, srcip, dstip, srcport, dstport, protocol):
        self.srcip = srcip
        self.dstip = dstip
        self.srcport = srcport
        self.dstport = dstport
        self.protocol = protocol

    def __hash__(self):
        return hash((self.srcip, self.dstip, self.srcport, self.dstport, self.protocol))

    def __eq__(self, other):
        return (self.srcip, self.srcport, self.dstip, self.dstport, self.protocol) == (other.srcip, other.srcport, other.dstip, other.dstport, other.protocol)
    
    def reverse(self):
        return flowKey(self.dstip, self.srcip, self.dstport, self.srcport, self.protocol)

flows = defaultdict(lambda: {
    'start_time': None,
    'last_time': None,
    'srcbytes': 0,
    'dstbytes': 0,
    'srcpkts': 0,
    'dstpkts': 0,
    'srcseq': None,
    'dstseq': None,
})

stop_event = threading.Event()
completed_flows = []

def get_flow_key(pkt):
    if IP not in pkt:
        return None
    ip = pkt[IP]
    protocol = ip.proto
    if protocol == 6 and TCP in pkt:
        l4 = pkt[TCP]
    elif protocol == 17 and UDP in pkt:
        l4 = pkt[UDP]
    else:
        return None
    return flowKey(ip.src, ip.dst, l4.sport, l4.dport, protocol)

def process_packet(pkt):
    if IP not in pkt:
        return
    key = get_flow_key(pkt)
    if not key:
        return
    now = datetime.datetime.now()
    
    if key in flows.keys():
        flow = flows[key]
        flow['last_time'] = now
        length = len(pkt)
        flow['srcbytes'] += length
        flow['srcpkts'] += 1
    elif key.reverse() in flows.keys():
        flow = flows[key.reverse()]
        flow['last_time'] = now
        length = len(pkt)
        flow['dstbytes'] += length
        flow['dstpkts'] += 1
        if TCP in pkt and flow['dstseq'] is None:
            flow['dstseq'] = pkt[TCP].seq
    else:
        flow = flows[key]
        flow['start_time'] = now
        flow['last_time'] = now
        length = len(pkt)
        flow['srcbytes'] += length
        flow['srcpkts'] += 1
        if TCP in pkt:
            flow['srcseq'] = pkt[TCP].seq

def flow_timeout_checker():
    start_time = datetime.datetime.now()
    while not stop_event.is_set():
        now = datetime.datetime.now()
        to_remove = []
        for key, flow in list(flows.items()):
            if (now - flow['last_time']).total_seconds() > FLOW_TIMEOUT:
                duration = (flow['last_time'] - flow['start_time']).total_seconds()
                srceff = (flow['srcbytes'] * 8 / duration) if duration > 0 else 0
                dsteff = (flow['dstbytes'] * 8 / duration) if duration > 0 else 0

                completed_flows.append({
                    'timeStamp': (flow['start_time'] - start_time).total_seconds(),
                    'srcip': key.srcip,
                    'dstip': key.dstip,
                    'srcport': key.srcport,
                    'dstport': key.dstport,
                    'protocol': '1' if key.protocol == 6 else '0',
                    'duration': round(duration, 6),
                    'srcbytes': flow['srcbytes'],
                    'dstbytes': flow['dstbytes'],
                    'srceff': round(srceff, 6),
                    'dsteff': round(dsteff, 6),
                    'srcpkts': flow['srcpkts'],
                    'dstpkts': flow['dstpkts'],
                    'srcseq': flow['srcseq'] if flow['srcseq'] is not None else '0',
                    'dstseq': flow['dstseq'] if flow['dstseq'] is not None else '0',
                    #'attack_catagory': '',  # default empty
                })
                to_remove.append(key)
        for key in to_remove:
            del flows[key]
        time.sleep(3)

def drop_flows(num):
    if num > len(completed_flows):
        return None
    data = completed_flows[:num].copy()
    completed_flows[:] = completed_flows[num:]
    return data

def save_flows_to_csv(completed_flows=completed_flows):
    df = pd.DataFrame(completed_flows)
    recordPath = 'local_data_set'
    recordPath += datetime.datetime.now().strftime("\\flow_%Y%m%d%H%M%S.csv")
    df.to_csv(recordPath, index=False)
    print(f"已儲存至{recordPath}")

def sniff_flow(record_time=RECORD_TIME):
    timeout_thread = threading.Thread(target=flow_timeout_checker, daemon=True)
    timeout_thread.start()
    print("開始擷取封包...")
    sniff(iface=get_if_list(),filter="udp or tcp",prn=process_packet, store=0, timeout=record_time)
    stop_event.set()
    timeout_thread.join()
    
if __name__ == "__main__":
    sniff_flow()
    save_flows_to_csv(completed_flows)