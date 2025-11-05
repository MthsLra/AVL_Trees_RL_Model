import torch
import torchrl.envs as envBase
import copy 
import os.path
import pickle


class BalancingTreeRL(envBase):

    def __init__(self,reuse_per_tree = 10, **kwargs):
        super.__init__()
        self.reuse_per_tree = reuse_per_tree
        self.episode_counter = 0
        self.tree_array = self._load_trees("dataset_trees.pkl")
        self.tree = None
        self.tree_index = 0
        self.og_tree = None
        self.reset()
    
    def _load_trees(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            print(data)
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
    
    def step(self, action):
        self.tree = self.apply_action(self.tree, action)

        reward = self.get_rewards(self.tree)
        obs = self.vectorize_tree(self.tree)
        done = self.is_done(self.tree)

        return 
    
    def apply_action(self, tree, action):
        return 
    
    def get_rewards(self, tree):
        return 
    
    def vectorize_tree(self, tree):
        return 
    
    def is_done(self, tree):
        return
    
