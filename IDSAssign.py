from numpy import mean, std
from FileUtils import readDictFromJson, saveDictToJson
from Network import FEALENGTH, PADDINGC, FEATOTALLEN

def identifyAllRepeatId(fingerprints):
    repeatIdMaps = dict()
    repeatIdMaps[True] = list()
    repeatIdMaps[False] = list()
    res = list()
    for c_fingerprint in fingerprints:
        c_id = {
            True: -1,
            False: -1,
        }
        for dir in [True, False]:
            cur_dir_fp = c_fingerprint[dir]
            if cur_dir_fp[0] == -1:
                continue
            if cur_dir_fp[0] not in repeatIdMaps[dir]:
                repeatIdMaps[dir].append(cur_dir_fp[0])
                c_id[dir] = len(repeatIdMaps[dir]) - 1
            else:
                c_id[dir] = repeatIdMaps[dir].index(cur_dir_fp[0])
        res.append(c_id)
    return res

def identifyLocalId(fingerprints):
    fingMap = {
        True: dict(),
        False: dict(),
    }
    res = list()
    for fingerprint in fingerprints:
        c_ids = {
            True: list(),
            False: list(),
        }
        for segs in fingerprint:
            fp_dir = tuple(segs[0])
            dir = segs[1]
            if fp_dir not in fingMap[dir]:
                fingMap[dir][fp_dir] = len(fingMap[dir])
            c_ids[dir].append(fingMap[dir][fp_dir])
        res.append(c_ids)
    return res


def identifyPktsId(fingerprints):
    pktIDMaps = {
        True: dict(),
        False: dict(),
    }
    res = list()
    for fingerprint in fingerprints:
        c_id = {
            True: -1,
            False: -1,
        }
        for dir in [True, False]:
            cur_dir_fp = fingerprint[dir]
            keys = tuple(set(cur_dir_fp.keys()))
            if keys not in pktIDMaps[dir]:
                pktIDMaps[dir][keys] = len(pktIDMaps[dir])
            c_id[dir] = pktIDMaps[dir][keys]
        res.append(c_id)
    return res


def identifyCircleId(fingerprints):
    idMaps = dict()
    res = list()
    for fingerprint in fingerprints:
        c_id = list()
        for keys in fingerprint.keys():
            if keys not in idMaps:
                idMaps[keys] = len(idMaps)
            c_id.append(idMaps[keys])
        res.append(c_id)
    return res


def identifyRepeatId(fingerprints):
    idMaps = {
        True: dict(),
        False: dict(),
    }
    res = list()
    for fingerprint in fingerprints:
        c_id = {
            True: list(),
            False: list(),
        }
        for dir in [True, False]:
            cur_dir_fp = fingerprint[dir]
            for keys in cur_dir_fp.keys():
                if keys not in idMaps[dir]:
                    idMaps[dir][keys] = len(idMaps[dir])
                c_id[dir].append(idMaps[dir][keys])
        res.append(c_id)
    return res


def identifyServics(fingerprints):
    portdatas = dict()
    for k, v in fingerprints.items():
        if k[3] not in portdatas:
            portdatas[k[3]] = dict()
        portdatas[k[3]][k] = v
    res = dict()
    for port, portdata in portdatas.items():
        p_flowids = [id for id, _ in portdata.items()]
        p_allrepeats = [item["allrepeat"] for _, item in portdata.items()]
        p_locals = [item["local"] for _, item in portdata.items()]
        p_pkts = [item["packet"] for _, item in portdata.items()]
        p_repeats = [item["repeat"] for _, item in portdata.items()]
        p_circles = [item["circle"] for _, item in portdata.items()]
        id_allrepeat = identifyAllRepeatId(p_allrepeats)
        id_local = identifyLocalId(p_locals)
        id_packet = identifyPktsId(p_pkts)
        id_repeat = identifyRepeatId(p_repeats)
        id_circle = identifyCircleId(p_circles)
        for flowid, al, lo, pa, re, ci in zip(p_flowids, id_allrepeat, id_local, id_packet, id_repeat, id_circle):
            res[flowid] = dict()
            res[flowid]["allrepeat"] = al
            res[flowid]["local"] = lo
            res[flowid]["packet"] = pa
            res[flowid]["repeat"] = re
            res[flowid]["circle"] = ci
    return res


def createNumIds(fingerprints):
    return [fingerprints["allrepeat"][True], fingerprints["allrepeat"][False], fingerprints["local"][True],
            fingerprints["local"][False], fingerprints["packet"][True], fingerprints["packet"][False],
            fingerprints["repeat"][True], fingerprints["repeat"][False], fingerprints["circle"]]


def getDataset(filepath):
    fpids = readDictFromJson(filepath)
    res = dict()
    for flowid, flow in fpids.items():
        srcip = eval(flowid)[0]
        if srcip not in res:
            res[srcip] = dict()
        res[srcip][flowid] = flow
    return res

def padding(fea):
    res = list()
    for i in range(len(fea)):
        if isinstance(fea[i], list):
            if len(fea[i]) < FEALENGTH:
                fea[i] = fea[i] + [PADDINGC] * (FEALENGTH - len(fea[i]))
            else:
                fea[i] = fea[i][:FEALENGTH]
            res.extend(fea[i])
        else:
            res.append(fea[i])
    totallen = 0
    for item in res:
        if isinstance(item, list):
            totallen += len(item)
        else:
            totallen += 1

    assert totallen == FEATOTALLEN
    return res


def saveIdWithInfo(flows, fpids, filepath):
    res = dict()
    for flowid, flow in flows.items():
        res[str(flowid)] = {
            "ids": padding([flowid[3]] + createNumIds(fpids[flowid])),
            "rawdata": flow
        }
    saveDictToJson(res, filepath)


def filterCommonIps(data, meanDelta=10, stdDelta=10, topnum=3):
    res = list()
    staticData = dict()
    for flowid, flowdata in data.items():
        flowid = eval(flowid)
        key = (flowid[2], flowid[3])
        if key not in staticData:
            staticData[key] = 1
        else:
            staticData[key] += 1
    staticData = sorted(staticData.items(), key=lambda x: x[1], reverse=True)[:topnum]
    tsData = dict()
    for flowid, flowdata in data.items():
        flowid = eval(flowid)
        key = (flowid[2], flowid[3])
        if key not in tsData:
            tsData[key] = list()
        timestamp = flowdata["rawdata"][0]["timestamp"]
        tsData[key].append(timestamp)
    for dpip in tsData.keys():
        tsdataip = tsData[dpip]
        tsData[dpip] = [tsdataip[i + 1] - tsdataip[i] for i in range(len(tsdataip) - 1)]
    for dpip, dpnum in staticData:
        if mean(tsData[dpip]) <= meanDelta and std(tsData[dpip]) <= stdDelta:
            res.append(dpip)
    return res


def deleteDataFromDatasetByDstIpAndPort(data, filterkeys):
    for flowid in list(data.keys()):
        flowid = eval(flowid)
        key = (flowid[2], flowid[3])
        if key in filterkeys:
            del data[str(flowid)]

