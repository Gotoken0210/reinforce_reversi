from abc import ABC
from abc import abstractclassmethod
from reversi.board import ReversiBoard

import random

from memory import Memory

from qFunction import QFunction
from reversi.player import Player

import numpy as np
import torch


class DQNPlayer(Player):
    # reversi player uses DQN
    def __init__(self, name, memory,
                 reward_win=1., reward_draw=0., reward_lose=-1.,
                 eps=1, eps_decrease=0.01, eps_min=0.01, alpha=0, print_player=False, beta=0.7, beta_increase=1/10000, use_IS=True, PER=False, batch_size=64, cond=0):
        self.name = name
        self.memory = memory
        self.reward_win = reward_win  # 勝利時報酬
        self.reward_draw = reward_draw  # 引き分け時報酬
        self.reward_lose = reward_lose  # 敗北時報酬
        self.eps = eps  # epsilon(ランダム行動確率)
        self.eps_decrease = eps_decrease  # epsの下がり幅
        self.eps_min = eps_min  # epsの最小値
        self.alpha = alpha  # prioritized experience replay のパラメータα
        self.print_player = print_player  # 表示の切り替え
        self.beta = beta  # beta(importance samplingで使用するハイパラ)
        self.beta_increase = beta_increase  # betaの増加幅
        self.use_IS = use_IS  # importance samplingの使用
        self.PER = PER  # prioritized experience replay
        self.batch_size = batch_size
        self.cond = cond  # まとめて回す際の条件指定

        self.q_function = QFunction(self.memory, gamma=0.95, alpha=self.alpha, rank_mode=True,
                                    beta=self.beta, beta_increase=self.beta_increase, use_IS=self.use_IS, PER=self.PER,  batch_size=self.batch_size, cond=self.cond)  # brain

        self.s_last = None
        self.a_last = None

        self.record = []

    # translate board from string to np.ndarray(1d)
    def str_board_to_list_board(self, str_board):
        list_board = np.zeros(64)
        for i in range(8):
            for j in range(8):
                if str_board[i * 9 + j] == '2':
                    list_board[i * 8 + j] = -1
                else:
                    list_board[i * 8 + j] = int(str_board[i * 9 + j])
        return list_board

    # def action(self, board):
    def action(self, state, possible_hand):
        if self.color == "b":
            str_maketrans = {'-': '0', 'b': '1', 'w': '2'}
        elif self.color == "w":
            str_maketrans = {'-': '0', 'b': '2', 'w': '1'}

        list_board = self.str_board_to_list_board(state.split(",")[0].translate(
            str.maketrans(str_maketrans)))

        pos_hand_list = possible_hand.split(",")
        # pass確定の時
        if pos_hand_list == ["0"]:
            if self.print_player:
                print("select pass")
            return "{}_{}".format(self.color, 0)

        if np.random.random() < self.eps:
            # epsilon-greedy. ランダムな行動
            a_board = int(
                pos_hand_list[np.random.randint(0, len(pos_hand_list))])
            a = a_board - 1  # playerクラスでのインデックスaに対応するboardに渡す値はa+1なため
            if self.print_player:
                print("select a_board({}) by epsilon-greedy".format(a_board))

        else:
            # NNを用いた行動選択
            self.q_function.main_network.eval()

            # Q値が大きくてかつ盤面が空いている部分へのaction
            with torch.no_grad():
                s = torch.unsqueeze(torch.from_numpy(list_board).type(
                    torch.FloatTensor), 0)

                # Q値昇順にindex(a)をソートして上からemptyチェック
                q_s_a_values = self.q_function.main_network(s).detach()
                q_s_a_desc_ind = q_s_a_values.sort(descending=True)

                a_ind = 0
                if self.print_player:
                    print("pos_hand:", pos_hand_list)
                while True:
                    a = int(q_s_a_desc_ind.indices[0][a_ind])
                    # playerクラスでのインデックスaに対応するboardに渡す値はa+1なため
                    a_board = a + 1
                    if self.print_player:
                        print("select a_board({})".format(a_board))
                    if str(a_board) in pos_hand_list:
                        break
                    else:
                        a_ind += 1

        # memorise state and action
        if self.s_last is not None:
            self.memory.append(self.s_last, self.a_last,
                               list_board, 0, 0, None)
        self.s_last = list_board
        self.a_last = a

        return "{}_{}".format(self.color, a_board)

    def finalize(self, state, winner):

        list_board = self.str_board_to_list_board(state.split(",")[0].translate(
            str.maketrans({'-': '0',
                           'b': '1',
                           'w': '2'})))

        if winner == self.color:
            r = self.reward_win
            self.record.append(1)  # WIN=1
        elif winner == "d":
            if self.print_player:
                print("DRAW")
            r = self.reward_draw
            self.record.append(0)  # DROW = 0
        else:
            if self.print_player:
                print("DQN LOSE")
            r = self.reward_lose
            self.record.append(-1)  # LOSE = -1

        self.memory.append(self.s_last, self.a_last, list_board, r, 1, None)

        self.s_last = None
        self.a_last = None

        # epsilonをエピソードの終了ごとに更新
        if self.eps_min < self.eps:
            self.eps -= self.eps_decrease
