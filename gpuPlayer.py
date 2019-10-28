from abc import ABC
from abc import abstractclassmethod
from reversi.board import ReversiBoard

import random

from memory import Memory

from qFunction import QFunction
from qFunction import GPUQFunction
from reversi.player import Player

import numpy as np
import torch


class GPUPlayer(Player):
    #reversi player uses DQN and GPU
    def __init__(self, name, memory,
             reward_win=1., reward_draw=0., reward_lose=-1.,
             eps=1, eps_decrease = 0.01, eps_min = 0.01, print_player = False):
        self.name = name
        self.memory = memory
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_lose = reward_lose
        self.eps = eps
        self.eps_decrease = eps_decrease
        self.eps_min = eps_min
        self.print_player = print_player
        

        self.q_function = GPUQFunction(self.memory, gamma = 0.95) #brain

        self.s_last = None
        self.a_last = None

        self.record = []
    
    #translate board from string to np.ndarray(1d)
    def str_board_to_list_board(self, str_board):
        list_board = np.zeros(64)
        for i in range(8):
            for j in range(8):
                list_board[i * 8 + j] = int(str_board[i * 9 + j])
        return list_board
    
    #def action(self, board):
    def action(self, state, possible_hand):        
        list_board = self.str_board_to_list_board(state.split(",")[0].translate(
            str.maketrans({'-': '0', 
                           'b': '1', 
                           'w': '2'})))
        
        pos_hand_list = possible_hand.split(",")
        #pass確定の時
        if pos_hand_list == ["0"]:
            if self.print_player:
                print("select pass")
            return "{}_{}".format(self.color, 0)
        
        if np.random.random() < self.eps:
            # epsilon-greedy. ランダムな行動
            a_board = int(pos_hand_list[np.random.randint(0, len(pos_hand_list))])
            a = a_board - 1 #playerクラスでのインデックスaに対応するboardに渡す値はa+1なため(ややこしい)
            if self.print_player:
                print("select a_board({}) by epsilon-greedy".format(a_board)) 
        else:
            ## NNを用いた行動選択.
            self.q_function.model.eval()
            
            ## Q値が大きくてかつ盤面が空いている部分へのaction
            with torch.no_grad():
                s = torch.unsqueeze(torch.from_numpy(list_board).type(torch.FloatTensor), 0) #FloatTensorの意図?
                
                #gpu
                s = s.to(torch.device("cuda:0"))
                
                ## Q値昇順にindex(a)をソートして上からEMPTYチェック
                q_s_a_desc_ind = self.q_function.model(s).detach().sort(descending = True)
                a_ind = 0
                if self.print_player:
                    print("pos_hand:", pos_hand_list)
                while True:
                    a = int(q_s_a_desc_ind.indices[0][a_ind])
                    a_board = a + 1 #playerクラスでのインデックスaに対応するboardに渡す値はa+1なため(ややこしい)
                    if self.print_player:
                        print("select a_board({})".format(a_board)) 
                    if str(a_board) in pos_hand_list: 
                        break
                    else:
                        a_ind+=1
                
        # memorise state and action
        if self.s_last is not None:
            self.memory.append(self.s_last, self.a_last, list_board, 0, 0, None)
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
            self.record.append(1) #WIN=1
        elif winner == "d":
            if self.print_player:
                print("DRAW")
            r = self.reward_draw
            self.record.append(0) #DROW = 0
        else:
            if self.print_player:
                print("DQN LOSE")
            r = self.reward_lose
            self.record.append(-1) #LOSE = -1

        self.memory.append(self.s_last, self.a_last, list_board, r, 1, None)

        self.s_last = None
        self.a_last = None
        
        #epsilonをエピソードの終了ごとに更新
        if self.eps_min < self.eps:
            self.eps -= self.eps_decrease
            