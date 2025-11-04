import torch
import torchrl.envs as envBase
import copy 
import os.path
import json 

class BalancingTreeRL(envBase):

    def __init__(self,reuse_per_tree = 10, **kwargs):
        super.__init__()
        self.reuse_per_tree = reuse_per_tree
        self.episode_counter = 0
        self.tree_array = self._load_trees("dataset_trees.txt")
        self.tree = None
        self.tree_index = 0
        self.og_tree = None
        self.reset()
    
    def _load_trees(self, filename):
        with open(filename, "r") as f:
            data = [json.loads(line.strip()) for line in f.readlines()]
        return data
    
    
    
    def reset(self, **kwargs):
        
        # We setup a new tree and change it every 10 episodes 
        if self.tree is None and self.episode_counter % self.reuse_per_tree != 0 :
            self.tree_index = (self.tree_index + 1) % len(self.tree_array)
            self.tree = self.tree[self.tree_index]
            self.og_tree = copy.deep_copy(self.tree)
        
        # We keep using the same tree
        else :
            self.tree = copy.deepcopy(self.og_tree)
        
        return self._get_observations()