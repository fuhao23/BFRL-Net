import itertools
import math
import random
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam

from FileUtils import readDictFromJson, readConfig

configData=readConfig()
FEALENGTH = configData["FEALENGTH"]
PADDINGC = configData["PADDINGC"]
FEATOTALLEN = 1 + 2 + FEALENGTH * 2 + 2 + FEALENGTH * 3
state_dim = FEATOTALLEN
action_dim = state_dim
hidden_dim = FEATOTALLEN
entropy_rate = configData["entropy_rate"]
reward_div_rate = configData["reward_div_rate"]
netUpdate_polyak = configData["netUpdate_polyak"]
buffer_size = configData["buffer_size"]
printInfo = configData["printInfo"]
epochs = configData["epochs"]
score_winsize = configData["score_winsize"]
startNetAction_step = 8 * score_winsize
sampleUpdateMin = configData["sampleUpdateMin"]
sampleUpdateStep = configData["sampleUpdateStep"]
learning_rate_actor = configData["learning_rate_actor"]
learning_rate_critic = configData["learning_rate_critic"]
batch_size = configData["batch_size"]
stateSize = score_winsize * 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            self.encoding[:, 1::2] = torch.cos(position * div_term[:-1])
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        expend = False
        if x.shape[-2] == 1:
            return x
        if len(x.shape) == 2:
            expend = True
            x = x.unsqueeze(0)
        totalFlowlen = x.size(-2) - 1
        splitEle = list()
        for i in range(totalFlowlen):
            x_f = torch.index_select(x, dim=-2, index=torch.tensor([i]).to(device))
            encode_f = x_f + self.encoding[:, 0, :].to(device)
            splitEle.append(encode_f)
        splitEle.append(
            torch.index_select(x, dim=-2, index=torch.tensor([totalFlowlen - 1]).to(device)).to(
                device) + torch.index_select(self.encoding,
                                             dim=-2,
                                             index=torch.tensor(
                                                 [1]).to(device)))
        x = torch.cat(splitEle, dim=-2)
        if expend:
            x = x.squeeze()
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=255):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.att_q = torch.nn.Linear(state_dim, hidden_dim)
        self.att_k = torch.nn.Linear(state_dim, hidden_dim)
        self.conv1d = nn.Conv1d(in_channels=FEATOTALLEN, out_channels=FEATOTALLEN, kernel_size=3, padding=1)
        self.activate = nn.ReLU()
        self.batch_norm_q = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_k = nn.BatchNorm1d(hidden_dim)
        fea_d_model = FEATOTALLEN
        fea_max_len = 2
        self.pos_encoder = PositionalEncoding(fea_d_model, fea_max_len).to(device)
        self.initParams()

    def initParams(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0, std=1)

    def forward(self, state, deterministic=False):
        state = self.pos_encoder(state)
        expand = False
        if len(state.shape) == 2:
            expand = True
        if expand:
            state = state.unsqueeze(0)
        state = state.transpose(-1, -2)
        state = self.activate(self.conv1d(state))
        state = state.transpose(-1, -2)
        if expand:
            state = state.squeeze(0)

        state_q = self.att_q(state)
        state_k = self.att_k(state)
        if state_q.shape[0]>1:
            if len(state_q.shape)==3:
                state_q = state_q.view(-1,self.hidden_dim)
                state_k = state_k.view(-1,self.hidden_dim)
            state_q = self.batch_norm_q(state_q)
            state_k = self.batch_norm_k(state_k)
        if not expand:
            state_q = state_q.view(batch_size,-1, self.hidden_dim)
            state_k = state_k.view(batch_size,-1, self.hidden_dim)

        state_k = torch.transpose(state_k, -1, -2)
        att = torch.matmul(state_q, state_k)[:, -1] / math.sqrt(state.shape[1])
        action_softmax_p = torch.softmax(att, dim=-1)
        if deterministic:
            action_p, action_index = torch.max(action_softmax_p, dim=-1)
        else:
            action_index_pos = torch.multinomial(action_softmax_p, 1)
            action_index = action_index_pos.squeeze(-1)
            action_p = torch.gather(action_softmax_p, -1, action_index_pos).squeeze(-1)
        return action_index, action_p


class CriticNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=255):
        super(CriticNet, self).__init__()
        hidden_dim = 1
        self.ln1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln3 = nn.Linear(hidden_dim * (stateSize + 1), stateSize + 1)
        self.ln4 = nn.Linear(stateSize + 1, 1)
        self.initParams()

    def initParams(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.normal_(param, mean=0, std=0.01)

    def forward(self, state, action):
        s_a = torch.cat((state, action.unsqueeze(-2) + 1), dim=-2)
        rew = self.relu(self.ln1(s_a))
        rew = rew.view(batch_size, -1)
        rew = self.relu(self.ln3(rew))
        rew = self.ln4(rew)
        return rew


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, hidden_dim=hidden_dim)
        self.critic1 = CriticNet(state_dim, act_dim, hidden_dim=hidden_dim)
        self.critic2 = CriticNet(state_dim, act_dim, hidden_dim=hidden_dim)
    def act(self, state, deterministic=False):
        with torch.no_grad():
            actor, _ = self.actor(state, deterministic)
            return actor.cpu().numpy()


# -----------------数据相关--------------------
class ReplayBuffer:
    def __init__(self, state_dim, size):
        self.state_buf = np.zeros((size, stateSize, state_dim), dtype=np.float32)
        self.state2_buf = np.zeros((size, stateSize, state_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.maxsize = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = state
        self.state2_buf[self.ptr] = next_state
        self.rew_buf[self.ptr] = reward
        self.act_buf[self.ptr] = action
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.maxsize
        self.size = min(self.size + 1, self.maxsize)

    def sample_batch(self, batch_size):
        idxs = np.arange(self.size + self.maxsize, self.size + self.maxsize - batch_size, -1) % self.maxsize

        batch = dict(state=self.state_buf[idxs],
                     state_next=self.state2_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class SAC:
    def __init__(self, dataset, state_dim, act_dim, hidden_dim=32):
        self.dataset = dataset
        self.datasetLen = len(dataset)
        self.normalizeDataset()
        self.traffic_score_table = dict()
        self.traffic_score_ids_table = dict()
        self.calTrafficScore()
        self.setSeed()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.actor_critic = ActorCritic(state_dim=state_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
        self.actor_critic_target = deepcopy(self.actor_critic)
        self.freezeTargetAC()
        self.critic_params = itertools.chain(self.actor_critic.critic1.parameters(),
                                             self.actor_critic.critic2.parameters())
        self.replay_buffer = ReplayBuffer(state_dim=state_dim, size=buffer_size)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = Adam(self.critic_params, lr=learning_rate_critic)

    def normalizeDataset(self):
        maxindex = [-1 for _ in range(len(list(self.dataset.values())[0]["ids"]))]
        for flowid in list(self.dataset.keys()):
            flowids = self.dataset[flowid]["ids"]
            flowids_new = [item + 1 for item in flowids]
            self.dataset[flowid]["ids"] = flowids_new
            for i in range(len(maxindex)):
                if flowids_new[i] > maxindex[i]:
                    maxindex[i] = flowids_new[i]
        for flowid in list(self.dataset.keys()):
            flowids = self.dataset[flowid]["ids"]
            for i in range(len(maxindex)):
                flowids[i] /= maxindex[i]

    def freezeTargetAC(self):
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

    def setSeed(self, seed=20240511):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def countVars(self, module):
        return sum([np.prod(p.shape) for p in module.parameters()])

    def getTotalVars(self):
        return tuple(
            self.countVars(module) for module in
            [self.actor_critic.actor, self.actor_critic.critic1, self.actor_critic.critic2])

    def calTrafficScore(self, winsize=score_winsize):
        for i in range(winsize, self.datasetLen + 1):
            data = list(self.dataset.values())[i - winsize:i]
            if i == winsize:
                for i_1 in range(winsize):
                    for i_2 in range(i_1 + 1, winsize):
                        key = str(data[i_1]["ids"] + data[i_2]["ids"])
                        if key in self.traffic_score_table:
                            self.traffic_score_table[key] += 1
                        else:
                            self.traffic_score_table[key] = 1
            else:
                for i_1 in range(winsize - 1):
                    key = str(data[i_1]["ids"] + data[-1]["ids"])
                    if key in self.traffic_score_table:
                        self.traffic_score_table[key] += 1
                    else:
                        self.traffic_score_table[key] = 1
        for keys, score in self.traffic_score_table.items():
            keys_list = eval(keys)
            indexnew = tuple(keys_list[0:FEATOTALLEN])
            indexAf = tuple(keys_list[FEATOTALLEN:])
            if indexnew not in self.traffic_score_ids_table:
                self.traffic_score_ids_table[indexnew] = dict()
            self.traffic_score_ids_table[indexnew][indexAf] = score
    def calLossCritic(self, data):
        state, action_index, reward, state_next, done = data["state"].to(device), torch.as_tensor(data["action"]).to(
            device), data["reward"].to(device), \
            data[
                "state_next"].to(device), \
            data["done"].to(device),
        action_index = action_index.to(torch.long)
        action = torch.gather(state, dim=1, index=action_index.view(-1, 1, 1).expand(-1, 1, state.shape[-1])).squeeze(
            -2)
        q1 = self.actor_critic.critic1(state, action)
        q2 = self.actor_critic.critic2(state, action)
        with torch.no_grad():
            action_next_index, action_p_next = self.actor_critic.actor(state_next)
            action_next = torch.gather(state_next, dim=1,
                                       index=action_next_index.view(-1, 1, 1).expand(-1, 1, state.shape[-1])).squeeze(
                -2)
            q1_pi_targ = self.actor_critic_target.critic1(state_next, action_next)
            q2_pi_targ = self.actor_critic_target.critic2(state_next, action_next)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            bellman_value = reward + reward_div_rate * (1 - done) * (q_pi_targ - entropy_rate * action_p_next)

        loss_q1 = ((q1 - bellman_value) ** 2).mean()
        loss_q2 = ((q2 - bellman_value) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q

    def get_action(self, state, deterministic=False):
        return self.actor_critic.act(torch.as_tensor(state, dtype=torch.float32).to(device),
                                     deterministic=deterministic)

    def getRandomAction(self, state):
        return random.choice([i for i in range(state.shape[-2])])

    def act(self, traffic_cls, actionindex, data):
        actionindex = int(actionindex)
        if actionindex == len(traffic_cls):
            traffic_cls_next = np.append(traffic_cls, [data], 0)
            return traffic_cls_next
        else:
            traffic_cls[actionindex] = data
            return traffic_cls

    def groupAct(self, traffic_group, traffic_group_infos, actionindex, data):
        actionindex = int(actionindex)
        if actionindex == len(traffic_group):
            traffic_group.append([data[1]["ids"]])
            traffic_group_infos.append([(data[0], data[1]["rawdata"], data[2])])
        else:
            traffic_group[actionindex].append(data[1]["ids"])
            traffic_group_infos[actionindex].append((data[0], data[1]["rawdata"], data[2]))

    def calActReward(self, traffic_cls, actionindex, data):
        feasDimensions = [2, 2 * FEALENGTH, 2, 2 * FEALENGTH, 1 * FEALENGTH]

        def judgeScore(feas):
            feas1 = feas[0]
            feas2 = feas[1]
            res = 0
            curp = 1
            for segs in feasDimensions:
                segsPairNum = sum(
                    1 for i in range(curp, curp + segs) if feas1[i] != -1 and feas2[i] != -1 and feas1[i] == feas2[i])
                if segs > 2:
                    length = sum(1 for i in range(curp, curp + segs) if feas1[i] != -1)
                    res += segsPairNum / length if length != 0 else 0
                else:
                    res += segsPairNum / 2
                curp += segs
            res /= len(feasDimensions)
            return res
        if actionindex == len(traffic_cls):
            return 1
        leaf_fea = traffic_cls[actionindex]
        keys = tuple(leaf_fea.tolist() + data)
        assert len(keys) == state_dim + action_dim

        keys_indexs = tuple(keys[0:FEATOTALLEN])
        if keys_indexs not in self.traffic_score_ids_table:
            return -1
        else:
            total_score = -float('inf')
            total_sim= -float('inf')
            for score_keys, score in self.traffic_score_ids_table[keys_indexs].items():
                sim = judgeScore([keys, score_keys])
                if sim * score > total_score and sim > 0.8:
                    total_score = sim * score
                    total_sim = sim
            if total_score < 1:
                return -1
            else:
                return total_score

    def calGroupsScore(self, groups):
        score = 0
        for group in groups:
            if len(group) == 1:
                pass
            else:
                for i in range(len(group) - 1):
                    key = str(group[i] + group[i + 1])
                    if key in self.traffic_score_table:
                        score += self.traffic_score_table[key]
        return score

    def test_agent(self):
        test_groups = list()
        test_groups_infos = list()
        for flowindex, (flowid, flowdata) in enumerate(self.dataset.items()):
            test_state = [item[-1] for item in test_groups] + [flowdata["ids"]]
            action = self.get_action(test_state, deterministic=True)
            self.groupAct(test_groups, test_groups_infos, action, (flowid, flowdata, flowindex))
        reward_total = self.calGroupsScore(test_groups)
        return reward_total

    def predict_agent(self):
        delta = 8
        test_groups = list()
        test_group_infos = list()
        for flowindex, (flowid, flowdata) in enumerate(self.dataset.items()):
            test_state_cur = [item[-1] for item in test_groups]
            test_state_index = [item[-1][2] for item in test_group_infos]
            test_state_new = list()
            filter_To_act_table = dict()
            for i, item in enumerate(test_state_index):
                if abs(item - flowindex) < delta:
                    test_state_new.append(test_state_cur[i])
                    filter_To_act_table[len(filter_To_act_table)] = i
            test_state_new.append(flowdata["ids"])
            filter_To_act_table[len(filter_To_act_table)] = len(test_state_index)
            action = int(self.get_action(test_state_new, deterministic=True))
            action = filter_To_act_table[action]
            self.groupAct(test_groups, test_group_infos, action, (flowid, flowdata, flowindex))
        reward_total = self.calGroupsScore(test_groups)
        return reward_total, test_groups, test_group_infos

    def calLossActor(self, data):
        state = data["state"].to(device)
        action_index, action_p = self.actor_critic.actor(state)
        action = torch.gather(state, dim=1, index=action_index.view(-1, 1, 1).expand(-1, 1, state.shape[-1])).squeeze(
            -2)
        q1_pi = self.actor_critic.critic1(state, action)
        q2_pi = self.actor_critic.critic2(state, action)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (entropy_rate * action_p - q_pi).mean()
        return loss_pi

    def update(self, data):
        res = {
            "act_loss": 0,
            "critic_loss": 0
        }
        self.critic_optimizer.zero_grad()
        loss_critic = self.calLossCritic(data)
        res["critic_loss"] = loss_critic.cpu().detach().numpy()
        loss_critic.backward()
        self.critic_optimizer.step()
        for p in self.critic_params:
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        loss_action = self.calLossActor(data)
        res["act_loss"] = loss_action.cpu().detach().numpy()
        loss_action.backward()
        self.actor_optimizer.step()
        for p in self.critic_params:
            p.requires_grad = True
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                p_targ.data.mul_(netUpdate_polyak)
                p_targ.data.add_((1 - netUpdate_polyak) * p.data)
        return res

    def run(self, modelpath):
        best_r = 0
        for epoch in range(epochs):
            cls_state = None
            for dataset_i, flowdata in enumerate(self.dataset.items()):
                flowid = flowdata[1]
                if dataset_i == self.datasetLen - 1:
                    done = True
                else:
                    done = False
                if cls_state is None:
                    cls_state = np.array([flowid["ids"]], dtype=np.float32)
                    continue
                if len(cls_state) < stateSize - 1:
                    cls_state = np.append(cls_state, [flowid["ids"]], 0)
                    continue
                cls_state_net = np.append(cls_state, [flowid["ids"]], 0)

                if dataset_i > startNetAction_step:
                    action = self.get_action(cls_state_net, deterministic=False)
                else:
                    action = self.getRandomAction(cls_state_net)

                cls_state_next_raw = self.act(cls_state.copy(), action, flowid["ids"])
                reward = self.calActReward(cls_state, action, flowid["ids"])

                if len(cls_state_next_raw) == stateSize:
                    cls_state_next_storage = cls_state_next_raw[1:]
                else:
                    cls_state_next_storage = cls_state_next_raw
                cls_state = cls_state_next_storage
                if dataset_i + 1 >= self.datasetLen:
                    cls_state_next_storate = np.append(cls_state_next_storage, [flowid["ids"]], 0)
                else:
                    cls_state_next_storate = np.append(cls_state_next_storage,
                                                       [list(self.dataset.values())[dataset_i + 1]["ids"]], 0)
                # 存储经验
                self.replay_buffer.store(cls_state_net, action, reward, cls_state_next_storate, done)
                if dataset_i % sampleUpdateStep == 0 and dataset_i >= sampleUpdateMin:
                    updatestep = int(dataset_i / sampleUpdateStep)
                    if updatestep % printInfo == 0:
                        print("Updating network：", updatestep)
                    loss = [list(), list()]
                    for _ in range(sampleUpdateStep):
                        batch = self.replay_buffer.sample_batch(batch_size=batch_size)
                        update_info = self.update(batch)
                        loss[0].append(update_info["act_loss"])
                        loss[1].append(update_info["critic_loss"])
                    if updatestep % printInfo == 0:
                        print(f"Updated! actor_loss: {np.mean(loss[0])} , critic_loss: {np.mean(loss[1])}")

            print(f"Epoch:{epoch}/{epochs}  Testing：")
            time_test = time.time()
            test_reward = self.test_agent()
            time_testend = time.time()
            print("Test done，reward：", test_reward, "spend time：", time_testend - time_test)
            torch.save(self.actor_critic.state_dict(), modelpath)
            print(f"Model saved，{best_r}==>{test_reward}")
            best_r = test_reward


def tableInfoOfSac(dataset):
    winsize = 8
    datasetLen = len(dataset)
    traffic_score_table = dict()
    for i in range(winsize, datasetLen + 1):
        windata = list(dataset.items())[i - winsize:i]
        keys = [eval(item[0]) for item in windata]
        data = [i[1] for i in windata]
        if i == winsize:
            for i_1 in range(winsize):
                for i_2 in range(i_1 + 1, winsize):
                    key = str(keys[i_1][2] + "_" + keys[i_2][2])
                    if key in traffic_score_table:
                        traffic_score_table[key] += 1
                    else:
                        traffic_score_table[key] = 1
        else:
            for i_1 in range(winsize - 1):
                key = str(keys[i_1][2] + "_" + keys[-1][2])
                if key in traffic_score_table:
                    traffic_score_table[key] += 1
                else:
                    traffic_score_table[key] = 1
    traffic_score_table = sorted(traffic_score_table.items(), key=lambda x: x[1], reverse=True)
    for k, v in traffic_score_table[:50]:
        print(k, v)



