from sklearn.utils import murmurhash3_32 
import scipy
import numpy as np
import array
import math
import heapq
np.random.seed(101)
from random_number_generator import ConsistentRandomNumberGenerator as CRNG
import pdb
import torch

def Hfunction(m, seed):
    #if seed is None:
    #  seed = np.random.randint(0,100000)
    return lambda x : murmurhash3_32(key=x, seed=seed, positive=True) % m

def Gfunction(seed):
    #if seed is None:
    #  seed = np.random.randint(0,100000)
    return lambda x : np.sign(murmurhash3_32(key=x, seed=seed))


class TopKDs() :
    ''' This is a data structure for top k elements 
        To optimize find, insert, delete minimum and update operations
        we implement the datastructure by maintaining a hashmap (dictionary) 
        and heap. Note that O(log(n)) time is not possible by maintaining only 
        one of these
    '''
    def __init__(self, k):
        self.capacity = k;
        self.dictionary = {}
    def show(self):
        print(self.dictionary)
    def insert(self, ids , values):
        ids = ids.cpu()
        values = values.cpu()
        for idx in range(0, len(ids)):
          self.dictionary[ids[idx]] = values[idx]
        k_keys_sorted_by_values = heapq.nlargest(self.capacity, self.dictionary, key=self.dictionary.get)
        ndictionary = {key:self.dictionary[key] for key in k_keys_sorted_by_values}
        del self.dictionary
        self.dictionary = ndictionary
        
        
                    
    def getTop(self):
        return self.dictionary
   

class CountSketch() :
    def __init__(self,d, R, max_d, topK, device_id):
        ''' d: number of hash functions
            R: range of the hash function. i.e. the memory
                to be used for count sketch
        '''
        self.sketch_memory = torch.zeros((d, R))
        if device_id != -1:
            self.sketch_memory = self.sketch_memory.cuda(device_id)
        self.d = d
        self.R = R
        # we are going to use 3 universal hash functions.
        # a * x^2 + b * x + c
        self.rng = CRNG(101)
        random_numbers = self.rng.generate(self.d * 6) # 3 for h and 3 for g

        self.HA = torch.LongTensor(random_numbers[0:self.d])
        self.HB = torch.LongTensor(random_numbers[self.d:2*self.d])
        self.HC = torch.LongTensor(random_numbers[2*self.d:3*self.d])

        self.GA = torch.LongTensor(random_numbers[3*self.d:4*self.d])
        self.GB = torch.LongTensor(random_numbers[4*self.d:5*self.d])
        self.GC = torch.LongTensor(random_numbers[5*self.d:6*self.d])
      
        if device_id != -1:
            self.HA, self.HB, self.HC = self.HA.cuda(device_id), self.HB.cuda(device_id), self.HC.cuda(device_id)
            self.GA, self.GB, self.GC = self.GA.cuda(device_id), self.GB.cuda(device_id), self.GC.cuda(device_id)

        self.max_d = max_d
        self.topkds = None
        self.topK = topK
        if topK is not None:
            self.topkds = TopKDs(topK)
        

    def insert(self, indices, values, thold): 
        print("Insert")
        # make the 
        #pdb.set_trace()
        # TODO assumming batch size 1
        l = values.shape[1]
        entry_matrix = torch.matmul(values.reshape(l, 1), values) # N X N
        ix = torch.transpose(indices, 0, 1).repeat(1, l)
        jx = indices.repeat((l,1))
        id_matrix = torch.triu(ix * self.max_d + jx, diagonal=1)
        # TODO take triu
        #hash_location_matrices 
  
        for i in range(self.d):
          h_matrix = (self.HA[i] * id_matrix**2  + self.HB[i] * id_matrix + self.HC[i])%self.rng.big_prime %self.R
          g_matrix = torch.triu(torch.sign((self.GA[i] * id_matrix**2  + self.GB[i] * id_matrix + self.GC[i])%self.rng.big_prime %2 - 0.5), diagonal=1) # Here we do the triu
          flat_index = h_matrix.reshape(1,-1).squeeze()
          flat_values = torch.mul(g_matrix, entry_matrix).reshape(1,-1).squeeze()
          self.sketch_memory[i].scatter_add_(0, flat_index, flat_values)
    
        print("Insert Complete")

        # Insert the top K into the heap structure. Heap with CS setting is a bit unreliable. with 
        #if self.topkds is not None :
          #values = self.query(id_matrix) # values will be NxN

          #print("TOPK insertion Start")
          #pdb.set_trace()
          #values = torch.triu(values, diagonal=1) # just in case
          #values = torch.abs(values) # absolute value
          #values = values.reshape(1,-1).squeeze()
          #ids = id_matrix.reshape(1,-1).squeeze()
          #idx = torch.topk(values, k=self.topK)[1] # topK should be small like 1000
          #values = values[idx]
          #ids = ids[idx]
          #self.topkds.insert(ids, values)
        #print("TOPK insertion done")

    def query(self, id_matrix): 
        #pdb.set_trace()
        
        print("Query")
        vs = []
        for i in range(self.d):
          h_matrix = (self.HA[i] * id_matrix**2  + self.HB[i] * id_matrix + self.HC[i])%self.rng.big_prime %self.R
          g_matrix = torch.triu(torch.sign((self.GA[i] * id_matrix**2  + self.GB[i] * id_matrix + self.GC[i])%self.rng.big_prime %2 - 0.5), diagonal=1) # Here we do the triu
          v = torch.mul(self.sketch_memory[i][h_matrix], g_matrix)
          vs.append(v)
        V = torch.stack(vs)
        V = torch.sort(V)[0] # self.d x N x N
        print("Query Complete")
        if self.d %2 == 1:
            return V[self.d//2]
        else:
            return (V[self.d//2 - 1] + V[self.d//2])/2


