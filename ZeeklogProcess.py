import json
from collections import namedtuple
from typing import List, Dict

Packet = namedtuple("Packet",
                    ["uid", "srcip", "srcport", "dstip", "dstport", "applayerlen", "direction", "info", "timestamp"])

def readZeekLog(filepath: str):
    packets = list()
    with open(filepath, 'r', encoding="utf8") as f:
        for line in f:
            linedata = json.loads(line)
            uid = linedata["uid"]
            srcip = linedata["srcip"]
            srcport = linedata["srcport"]
            dstip = linedata["dstip"]
            dstport = linedata["dstport"]
            is_orig = linedata["is_orig"]
            timestamp = linedata["timestamp"]
            applayerlength = linedata["applayerlength"]
            if "appinfo" in linedata:
                packet = Packet(uid, srcip, srcport, dstip, dstport, applayerlength, is_orig, linedata["appinfo"],
                                timestamp)
            else:
                packet = Packet(uid, srcip, srcport, dstip, dstport, applayerlength, is_orig, None, timestamp)

            packets.append(packet)
    return packets

def aggregatePacketToFlow(packets: List[Packet]):
    uidflow = dict()
    for packet in packets:
        if packet.uid not in uidflow:
            uidflow[packet.uid] = list()
        uidflow[packet.uid].append(packet)
    ipflow = dict()
    for uid, flows in uidflow.items():
        keys = (flows[0].srcip, flows[0].srcport, flows[0].dstip, flows[0].dstport)
        if keys not in ipflow:
            ipflow[keys] = list()
        ipflow[keys].append(flows)
    for ipk in ipflow.keys():
        ipflow[ipk].sort(key=lambda pair: pair[0].timestamp)
        newdata = list()
        for uidflow in ipflow[ipk]:
            newdata.extend([{"applayerlength": packet.applayerlen, "direction": packet.direction, "info": packet.info,
                             "timestamp": packet.timestamp} for packet in uidflow])
        ipflow[ipk] = newdata
    return ipflow

def aggregateFlowDirToSrcip(flows):
    srcipdata = dict()
    for (srcip, srcport, dstip, dstport), flowdata in flows.items():
        if srcip not in srcipdata:
            srcipdata[srcip] = dict()
        srcipdata[srcip][(srcip, srcport, dstip, dstport)] = flowdata
    return srcipdata
def aggregateFlowDirToDstip(flows):

    srcipdata = dict()
    for (srcip, srcport, dstip, dstport), flowdata in flows.items():
        if dstip not in srcipdata:
            srcipdata[dstip] = dict()
        srcipdata[dstip][(srcip, srcport, dstip, dstport)] = flowdata
    return srcipdata

def aggregateFlowToDstipAndPort(flows):
    dstipdata = dict()
    for (srcip, srcport, dstip, dstport), flowdata in flows.items():
        if dstip not in dstipdata:
            dstipdata[dstip] = dict()
        if dstport not in dstipdata[dstip]:
            dstipdata[dstip][dstport] = dict()
        dstipdata[dstip][dstport][(srcip, srcport, dstip, dstport)] = flowdata
    return dstipdata

def aggregateFlowToSession(flows: Dict) -> Dict:
    ips = dict()
    for key, flowdata in flows.items():
        ip_key = (key[0], key[2])
        if ip_key not in ips:
            ips[ip_key] = dict()
        ips[ip_key][(key[1], key[3])] = flowdata
    return ips


def aggregateSessionToSrcIP(session: Dict) -> Dict:
    ipdata = dict()
    for (srcip, dstip), seesiondata in session.items():
        if srcip not in ipdata:
            ipdata[srcip] = dict()
        ipdata[srcip][dstip] = seesiondata
    return ipdata


def aggregateSessionToDstIP(session: Dict) -> Dict:
    ipdata = dict()
    for (srcip, dstip), seesiondata in session.items():
        if dstip not in ipdata:
            ipdata[dstip] = dict()
        ipdata[dstip][srcip] = seesiondata
    return ipdata


def aggregateFlowToService(flows: Dict) -> Dict:
    dstportdata = dict()
    for flow_key, flowdata in flows.items():
        dstport = flow_key[3]
        dstip = flow_key[2]
        if dstport not in dstportdata:
            dstportdata[dstport] = dict()
        if dstip not in dstportdata[dstport]:
            dstportdata[dstport][dstip] = list()
        dstportdata[dstport][dstip].append({flow_key: flowdata})
    return dstportdata

def printServices(services):
    for port, datas in services.items():
        print("------------------------------------")
        print(port)
        for dstip, dstipdata in datas.items():
            print("【", dstip, "】")
            for flowdata in dstipdata:
                print(". .. .. .. .. .. .. . .")
                for flowid, flow in flowdata.items():
                    print(flowid, [packet["applayerlength"] for packet in flow],
                          [packet["info"] for packet in flow if "info" in packet][:1], )
