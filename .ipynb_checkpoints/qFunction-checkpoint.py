import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np

#ネットワークモデル
class DQN(nn.Module):
    def __init__(self, n_in, n_out):
        super(DQN, self).__init__()
        n_mid = 128
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = torch.tanh(self.fc3(h2))
        return output

class DuelingDQN(nn.Module):
    def __init__(self, n_in, n_out):
        super(DuelingDQN, self).__init__()
        n_mid = 128
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        #ここからDueling
        self.fc3_adv = nn.Linear(n_mid, n_out) #advantage
        self.fc3_v = nn.Linear(n_mid, 1) #value
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        #ここからDueling
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1)) #advとの加算のためにexpand
        
        q = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        output = torch.tanh(q)
        return output
    
#ココカラ0910夜用
class DuelingDQN_2_5(nn.Module):
    def __init__(self, n_in, n_out):
        super(DuelingDQN_2_5, self).__init__()
        n_mid = 128
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc2_1 = nn.Linear(n_mid, n_mid)
        self.fc2_2 = nn.Linear(n_mid, n_mid)
        #ここからDueling
        self.fc3_adv = nn.Linear(n_mid, n_out) #advantage
        self.fc3_v = nn.Linear(n_mid, 1) #value
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2_1 = F.relu(self.fc2_1(h2))
        h2_2 = F.relu(self.fc2_2(h2_1))
        
        #ここからDueling
        adv = self.fc3_adv(h2_2)
        val = self.fc3_v(h2).expand(-1, adv.size(1)) #advとの加算のためにexpand
        
        q = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        output = torch.tanh(q)
        return output
    
class DuelingDQN_3_6(nn.Module):
    def __init__(self, n_in, n_out):
        super(DuelingDQN_3_6, self).__init__()
        n_mid = 256
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        #ここからDueling
        self.fc3_adv = nn.Linear(n_mid, n_out) #advantage
        self.fc3_v = nn.Linear(n_mid, 1) #value
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        #ここからDueling
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1)) #advとの加算のためにexpand
        
        q = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        output = torch.tanh(q)
        return output
    

class QFunction(object): #Q関数. Brainに相当.
    def __init__(self, memory, gamma, rank_mode, alpha, beta, beta_increase, use_IS, PER, batch_size, cond):
        self.num_states = 64
        self.num_actions = 64
        self.memory = memory
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.use_IS = use_IS
        self.rank_mode = rank_mode #prioritized experience replayをpi計算を順位ベースで
        self.loss_log = []
        self.ind_log = []
        self.batch_size = batch_size
        self.PER = PER
        
        self.cond = cond

        if self.cond == 1 or self.cond == 4:
            self.main_network = DuelingDQN(self.num_states, self.num_actions)
            self.target_network = DuelingDQN(self.num_states, self.num_actions)
            print(self.main_network)
        elif self.cond == 2 or self.cond == 5:
            self.main_network = DuelingDQN_2_5(self.num_states, self.num_actions)
            self.target_network = DuelingDQN_2_5(self.num_states, self.num_actions)
            print(self.main_network)
        elif self.cond == 3 or self.cond == 6:
            self.main_network = DuelingDQN_3_6(self.num_states, self.num_actions)
            self.target_network = DuelingDQN_3_6(self.num_states, self.num_actions)
            print(self.main_network)
        else:
            print("kasu")
        

        ##optimizerはAdamを使用
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=0.0001)
        
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def update_td(self, end): #endは基本 min(self.memory.counter, self.memory.size)
        #TD誤差の計算
        state_list_td = []
        action_list_td = []
        sdash_list_td = []
        reward_list_td = []
        end_list_td = []
        head = None
        
        for i in range(end):
            s, a, sdash, r, e, td = self.memory.read(i)
            state_list_td.append(s)
            action_list_td.append([a])
            sdash_list_td.append(sdash)
#             reward_list_td.append([r])
            reward_list_td.append(r)

            end_list_td.append([e])
            
        state_batch_td = torch.Tensor(state_list_td)
        action_batch_td = torch.LongTensor(action_list_td) ##整数の場合はLongTensor. gatherに渡すため.
        sdash_batch_td = torch.Tensor(sdash_list_td)
        reward_batch_td = torch.FloatTensor(reward_list_td)
        end_batch_td = torch.FloatTensor(end_list_td)
        
        self.main_network.eval()
        
        with torch.no_grad():
            q_s_a_values_td = self.main_network(state_batch_td).detach().gather(1, action_batch_td)
            q_s_a_values_td = q_s_a_values_td.view(q_s_a_values_td.size()[0])
            
            ## maxQ(s', a)を求める. e=1なら0
            max_q_sdash_a_values_td = torch.zeros(len(state_list_td))

            # max_q_sdash_a_valuesにmaxQ(s', a)の値を格納
            for idx, e in enumerate(end_list_td): #多分tensor活用した書き方できる…
                if e[0] == 0: #non_final
                    max_q_sdash_a_values_td[idx] = self.main_network(sdash_batch_td[idx]).max(0)[0].detach().item()
#                     print("max_q_sdash_a_values_td[idx]: ", max_q_sdash_a_values_td[idx])
        
        td = reward_batch_td + self.gamma * max_q_sdash_a_values_td - q_s_a_values_td
        td.abs_()
        
#         print("TD: {}".format(td))
        #memoryにTD誤差(の絶対値)書き込み
#         print("len td:{}".format(len(td)))
#         print("Write TD ({} to {})".format(0, len(td) - 1))
        
#         print("len(memory):{}".format(len(self.memory)))
        
        pos = self.memory.get_td_pos()
        for i in range(end):
            self.memory.memory[i, pos] = td[i].item()
#         print("memory:{}".format(self.memory.memory))
        
    def make_batch_and_update_weights(self):
        state_list = []
        action_list = []
        sdash_list = []
        reward_list = []
        end_list = []
        
        non_final_sdash_list = []
        
        
        weights = np.empty(self.batch_size, dtype='float32') #ISのための重み
        
        
        td_array = self.memory.memory[:min(self.memory.counter, len(self.memory)), self.memory.get_td_pos()]
#         print("len(memory):", len(self.memory), ", len(td_array):", len(td_array))
#         print("td_array:{}".format(td_array))

        td_array_sorted = np.sort(td_array)[::-1] #np.sort, np.argsortは昇順なため[::-1]
#         print("td_array_sorted:{}".format(td_array_sorted)) 

        td_array_sorted_index = np.argsort(td_array)[::-1]
#         print("td_array_sorted_index:{}".format(td_array_sorted_index))


        priority_sorted = np.power(1 / (np.arange(len(td_array_sorted_index))+1), self.alpha) #p_i
#         print("priority_sorted:{}".format(priority_sorted))

        priority_sorted = priority_sorted / np.sum(priority_sorted) #P_i
#         print("priority_sorted(normalized):{}".format(priority_sorted))

        priority_sorted_cumsum = np.cumsum(priority_sorted)
#         print("priority_sorted_cumsum:{}".format(priority_sorted_cumsum))

        sampled_ind = []
        thr_0 = 1 / self.batch_size
        thr = thr_0
        head = tail = 0
        for tail in range(len(priority_sorted_cumsum)):
            if thr < priority_sorted_cumsum[tail]:
#                 print("head:{}, tail:{}".format(head, tail))
#                 print("thr:{}, cumsum[tail]:{}".format(thr, priority_sorted_cumsum[tail]))
                if tail <= head + 1:
#                     print("-sample head {}(sorted_index[{}])(tail <= head + 1)\n".format(td_array_sorted_index[head], head))
                    thr += thr_0
                    sampled_ind.append(td_array_sorted_index[head])
                    head += 1
                else:
                    thr += thr_0
                    ind = np.random.randint(head, tail)
                    sampled_ind.append(td_array_sorted_index[np.random.randint(head, tail)])
#                     print("--sample {}(sorted_index[{}])(head +1 < tail)\n".format(sampled_ind[len(sampled_ind)-1], ind))
                    head = tail

        #データ足りてない → 最後のセグメントからサンプリングができていない
        if len(sampled_ind) < self.batch_size:
#             print(len(sampled_ind), "/", self.batch_size, "=>", "Lack")
            if head == tail:
                sampled_ind.append(head)
            else:
                sampled_ind.append(td_array_sorted_index[np.random.randint(head, tail)])
#             print(len(sampled_ind), "/", self.batch_size, "=>", "Lack?")
#         print("sampled_ind:{}".format(sampled_ind))

        ## priorityに従ってexperience replay → s, a, sdash, r, eに関してbatch_size行のtensor作成
#         for i in range(batch_size):
#             ind = np.random.randint(0, len(self.memory))

        total = np.sum(self.memory.memory[:, self.memory.get_td_pos()])
        for pos, ind in enumerate(sampled_ind):
            self.ind_log.append(ind)
            s, a, sdash, r, e, td = self.memory.read(ind)
            state_list.append(s)
            action_list.append([a])
            sdash_list.append(sdash)
#             reward_list.append([r])
            reward_list.append(r)

            end_list.append([e])
        
            #サンプリングしたデータについてISの重み計算
            if self.use_IS:
                priority_pos = priority_sorted[td_array_sorted_index[ind]]
                weights[pos] = (self.memory.size * priority_pos / total) ** (-self.beta)
        
        #weightsは1以下となるように
        weights /= weights.max()
        
        #batch作成
        self.state_batch = torch.Tensor(state_list)
        self.action_batch = torch.LongTensor(action_list) ##整数の場合はLongTensor. gatherに渡すため.
        self.sdash_batch = torch.Tensor(sdash_list)
        self.reward_batch = torch.FloatTensor(reward_list)
        self.end_batch = torch.FloatTensor(end_list)

        return weights        

    def make_batch(self):
        state_list = []
        action_list = []
        sdash_list = []
        reward_list = []
        end_list = []
        
        non_final_sdash_list = []
        
        for i in range(self.batch_size):
            ind = np.random.randint(0, len(self.memory))
            self.ind_log.append(ind)
            s, a, sdash, r, e, td = self.memory.read(ind)
            state_list.append(s)
            action_list.append([a])
            sdash_list.append(sdash)
#             reward_list.append([r])
            reward_list.append(r)

            end_list.append([e])
            if e == 0:
                non_final_sdash_list.append(sdash)
        
        self.state_batch = torch.Tensor(state_list)
        self.action_batch = torch.LongTensor(action_list) ##整数の場合はLongTensor. gatherに渡すため.
        self.sdash_batch = torch.Tensor(sdash_list)
        self.reward_batch = torch.FloatTensor(reward_list)
        self.end_batch = torch.FloatTensor(end_list)
        
        self.non_final_sdash_batch = torch.Tensor(non_final_sdash_list)
        
    
    def train(self):  
        
        if self.PER: #prioritized experience replayを使用            
            self.update_td(min(self.memory.counter, self.memory.size))
            
            weights = self.make_batch_and_update_weights()
            
            #最終的に1になるまでbetaを更新
            if self.beta + self.beta_increase < 1:
                self.beta += self.beta_increase
            else: 
                self.beta = 1  
                
        else: #通常のexperience replayを使用
            self.make_batch()            
        
        ##　損失関数の計算
        self.main_network.eval()
        self.target_network.eval()
        
        ## 損失関数中のQ^new_theta(s, a)
        q_s_a_values = self.main_network(self.state_batch).gather(1, self.action_batch)
        ## maxQ(s', a)を求める. e=1なら0
        max_q_sdash_a_values = torch.zeros(self.batch_size) #0で初期化

        # max_q_sdash_a_valuesにmaxQ(s', a)の値を格納
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor)
        
        #s'におけるQ値最大の行動(a_mを)main networkから求める

        non_final_mask = torch.tensor(tuple(map(lambda e: e == [0], self.end_batch.tolist())))

        a_m[non_final_mask] = self.main_network(self.non_final_sdash_batch).detach().max(1)[1]
        
        a_m_non_final = a_m[non_final_mask].view(-1, 1) # 最後でないものだけにして size →size*1
        
        max_q_sdash_a_values[non_final_mask] = self.target_network(self.non_final_sdash_batch).gather(1, a_m_non_final).detach().squeeze()
            
        ## 損失関数中のQ_theta(s, a)
        
        
#         print("reward_batch.size():", reward_batch.size())
#         print("max_q_sdash_a_values:", max_q_sdash_a_values)
#         print("max_q_sdash_a_values.size():", max_q_sdash_a_values.size())
        expected_q_s_a_values = self.reward_batch + self.gamma * max_q_sdash_a_values

        ## パラメータ更新
        self.main_network.train()

#         print("q_s_a_values.size()", q_s_a_values.size())
#         print("expected_q_s_a_values.size()", expected_q_s_a_values.size())
#         print("expected_q_s_a_values.unsqueeze(1).size()", expected_q_s_a_values.unsqueeze(1).size())
        
        if self.PER and self.use_IS:
            loss = self.IS_loss(q_s_a_values, expected_q_s_a_values.unsqueeze(1), weights)
        else:
            loss = F.smooth_l1_loss(q_s_a_values, expected_q_s_a_values.unsqueeze(1))
        
        self.loss_log.append(loss.item())
        
        ## 勾配リセット, 逆伝播, パラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def IS_loss(self, input, target, weights):
        if self.use_IS:
            loss = torch.abs(target - input) * torch.from_numpy(weights)
#             print("loss:{}".format(loss))
            return loss.mean()