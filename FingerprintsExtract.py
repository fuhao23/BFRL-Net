from collections import Counter
from FileUtils import readDictFromJson

def extractFlowByDir(flows, dirs):
    res = {
        True: list(),
        False: list(),
    }
    for index, dir in enumerate(dirs):
        res[dir].append(flows[index])
    return res

def extractRepreatFromList(data):
    if len(set(data)) == 1:
        return data[0], len(data)
    else:
        return -1, -1

def extractALLRepeatFingerprints(flows, dirs):
    flowByDir = extractFlowByDir(flows, dirs)
    return {
        True: extractRepreatFromList(flowByDir[True]),
        False: extractRepreatFromList(flowByDir[False]),
    }

def extractFlowSegsByLocalDir(flows, dirs):
    if len(flows) != len(dirs):
        return "error"
    res = list()
    cur_dir = dirs[0]
    cur_list = [flows[0]]
    for i, p_dir in enumerate(dirs):
        if i == 0:
            continue
        if p_dir == cur_dir:
            cur_list.append(flows[i])
        else:
            res.append(cur_list)
            cur_list = [flows[i]]
            cur_dir = p_dir
    res.append(cur_list)
    return res


def judgeSegPair(segs1, segs2):
    if set(segs1) == set(segs2):
        return True
    if sum(segs1) == sum(segs2):
        return True
    return False


def judgeSegPairBA(segs1, segs_p):
    for item in segs1:
        if item not in segs_p:
            return False
    return True


def extractLocalFingerprints(datas):
    if len(datas) == 1:
        return [list()]
    res = list()
    data_segs = [extractFlowSegsByLocalDir(item[0], item[1]) for item in datas] 
    caches = dict()
    for c_index, c_seg in enumerate(data_segs):
        c_datalen = len(c_seg)
        best_pair_num = 0
        best_pairs = list()
        pair_count = 0
        for p_index, p_seg in enumerate(data_segs):
            if p_index == c_index:
                continue
            if abs(p_index - c_index) > most_count:
                continue
            elif c_index > p_index:
                keys = str(p_index) + "_" + str(c_index)
                p_pair_items, p_pair_num = caches[keys]
                del caches[keys]
            else:
                p_datalen = len(p_seg)
                datalen = min(c_datalen, p_datalen)
                p_pair_num = 0
                p_pair_items = list()
                dir = True
                for i in range(datalen):
                    c_pkt = c_seg[i]
                    p_pkt = p_seg[i]
                    lazy_pairs = list()
                    index_after = i + 2
                    if index_after < datalen:
                        lazy_pairs.extend(p_seg[index_after])
                    index_before = i - 2
                    if index_before >= 0:
                        lazy_pairs.extend(p_seg[index_before])
                    lazy_pairs.extend(p_pkt)
                    if judgeSegPair(c_pkt, p_pkt):
                        p_pair_items.append([c_pkt, dir])
                        p_pair_num += len(p_pkt)
                    elif judgeSegPairBA(c_pkt, lazy_pairs):
                        p_pair_items.append([c_pkt, dir])
                        p_pair_num += len(p_pkt)
                    dir = not dir
                keys = str(c_index) + "_" + str(p_index)
                caches[keys] = (p_pair_items, p_pair_num)
            if p_pair_num > best_pair_num:
                best_pair_num = p_pair_num
                best_pairs = p_pair_items
            pair_count += 1
        res.append(best_pairs)
    return res


def countList(data):
    return dict(Counter(data))


def judgeCommonList(data, data_p):
    data_static = countList(data)
    data_p_static = countList(data_p)
    res = dict()
    for k, v in data_static.items():
        if k in data_p_static:
            pair_num = data_p_static[k]
            c_num = min(v, pair_num)
            res[k] = c_num
    res_num = sum([v for _, v in res.items()])
    return res, res_num


most_count = 20  


def extractPacketFingerprints(datas):
    res = list()
    if len(datas) == 1:
        res.append({
            True: dict(),
            False: dict()
        })
        return res

    data_dirs = [extractFlowByDir(item[0], item[1]) for item in datas]  
    cache = dict()
    for c_index, c_data in enumerate(data_dirs):
        c_res = {
            True: None,
            False: None,
        }
        for dir in [True, False]:
            best_data = dict()
            best_num = 0

            for p_index, p_data in enumerate(data_dirs):
                if abs(c_index - p_index) > most_count:
                    continue
                if c_index == p_index:
                    continue
                else:
                    c_data_dir = c_data[dir]
                    p_data_dir = p_data[dir]
                    list_data, list_num = judgeCommonList(c_data_dir, p_data_dir)
                    keys = str(c_index) + "_" + str(p_index) + str(dir)
                    cache[keys] = (list_data, list_num)
                if list_num > best_num:
                    best_num = list_num
                    best_data = list_data
            c_res[dir] = best_data
        res.append(c_res)
    return res


def extractRepeatFingerprints(flows, dirs):
    res = {
        True: dict(),
        False: dict(),
    }
    data = extractFlowByDir(flows, dirs)
    for dir in [True, False]:
        data_dir = countList(data[dir])
        for k, v in data_dir.items():
            if v > 1:
                res[dir][k] = v
    return res


def extractCircleFingerprints(flows, dirs):
    segdatas = extractFlowSegsByLocalDir(flows, dirs)
    badatas = [tuple(set(segdatas[i] + segdatas[i + 1])) for i in range(len(segdatas) - 1)]
    badatasList = countList(badatas)
    res = dict()
    for k, v in badatasList.items():
        if v > 1:
            res[k] = v
    return res


def extractFingerprintsFromFlow(flows):
    dstportFingers = dict()
    for dstipindex, (dstip, dstipdata) in enumerate(flows.items()):
        for dstport, dstportdata in dstipdata.items():
            localInputsdata = list()
            keys = list()
            for flowid_index, (flowid, flow) in enumerate(dstportdata.items()):
                flowpktlengths = [packet["applayerlength"] for packet in flow]
                flowpktdirs = [packet["direction"] for packet in flow]
                allrepeatFeas = extractALLRepeatFingerprints(flowpktlengths, flowpktdirs)
                repeatFeas = extractRepeatFingerprints(flowpktlengths, flowpktdirs)
                cirFeas = extractCircleFingerprints(flowpktlengths, flowpktdirs)
                dstportFingers[flowid] = {
                    "allrepeat": allrepeatFeas,
                    "repeat": repeatFeas,
                    "circle": cirFeas,
                }
                localInputsdata.append([flowpktlengths, flowpktdirs])
                keys.append(flowid)
            localFeas = extractLocalFingerprints(localInputsdata)
            packetFeas = extractPacketFingerprints(localInputsdata)
            for flowid, localFea, packetFea in zip(keys, localFeas, packetFeas):
                dstportFingers[flowid]["local"] = localFea
                dstportFingers[flowid]["packet"] = packetFea
    return dstportFingers


def scanDatasetInfo(filepath):
    data = readDictFromJson(filepath)
    for flowid, flowinfo in data.items():
        print(flowid)
        print([item["applayerlength"] for item in flowinfo["rawdata"]])
        print()
        print([item["info"] for item in flowinfo["rawdata"] if item["info"] is not None])


