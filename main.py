import torch
from FileUtils import readConfig
from FingerprintsExtract import extractFingerprintsFromFlow
from IDSAssign import getDataset, filterCommonIps, deleteDataFromDatasetByDstIpAndPort, identifyServics, saveIdWithInfo
from Network import SAC, action_dim
from ZeeklogProcess import readZeekLog, aggregatePacketToFlow, aggregateFlowToDstipAndPort

if __name__ == '__main__':
    """
    STEP assigns the corresponding operation:
    1、Extract fingerprint ID, save to file
    2、Training
    3、Test
    """
    step = 3
    if step == 1:
        zeekfilepath = "./data/20240902Features.log"  # Please replace with zeek log path
        idsFilepath = "./20240902Features.json"  # Please replace with the ID output path of the flows
        packets = readZeekLog(zeekfilepath)
        flows = aggregatePacketToFlow(packets)
        flows = sorted(flows.items(), key=lambda x: x[1][0]["timestamp"])
        flows = {item[0]: item[1] for item in flows}
        dstipdata = aggregateFlowToDstipAndPort(flows)
        feas = extractFingerprintsFromFlow(dstipdata)
        fpids = identifyServics(feas)
        saveIdWithInfo(flows, fpids, idsFilepath)

    elif step == 2 or step == 3:
        fpidFilpath = "./20240902Features.json"  #It needs to be the same as idsFilepath.
        hostip = readConfig()["hostip"]
        modelpath = "./20240902.pt"  # Model weighting path
        dataset = getDataset(fpidFilpath)
        dataset = dataset[hostip]
        filterDatas = filterCommonIps(dataset)
        deleteDataFromDatasetByDstIpAndPort(dataset, filterDatas)
        sac = SAC(dataset, state_dim=action_dim, act_dim=action_dim)
        if step == 2:
            sac.run(modelpath)
        elif step == 3:
            sac.actor_critic.load_state_dict(torch.load(modelpath))
            rew, res, res_infos = sac.predict_agent()
            step = 0
            for g, g_infos in zip(res, res_infos):
                print("Group number:", len(g))
                for item, iteminfo in zip(g, g_infos):
                    flowid = eval(iteminfo[0])
                    flowdatas = iteminfo[1]
                    flowindex = iteminfo[2]
                    print(flowindex, flowid)
                    print([i["info"] for i in flowdatas if i["info"] is not None and "name" in i["info"]])
                print("-------------------------" * 2)
