import numpy as np

class Memory(object):

    def __init__(self, size=64):
        self.size = size
        self.memory = np.empty((size, 132), dtype=np.float32) #64+1+64+1+1+1=132
        self.counter = 0

    def __len__(self):
        return min(self.size, self.counter)

    def read(self, ind):
        s = self.memory[ind, :64].astype(np.int32)
        a = self.memory[ind, 64].astype(np.int32)
        s_dash = self.memory[ind, 65:129].astype(np.int32)
        r = self.memory[ind, 129]
        e = self.memory[ind, 130]
        td = self.memory[ind, 131]
        return s, a, s_dash, r, e, td

    def write(self, ind, s, a, s_dash, r, e, td):
        self.memory[ind, :64] = s
        self.memory[ind, 64] = a
        self.memory[ind, 65:129] = s_dash
        self.memory[ind, 129] = r
        self.memory[ind, 130] = e
        self.memory[ind, 131] = td

    def append(self, s, a, s_dash, r, e, td):
        ind = self.counter % self.size
        self.write(ind, s, a, s_dash, r, e, td)
        self.counter += 1
        
    def get_td_pos(self):
        return 131 