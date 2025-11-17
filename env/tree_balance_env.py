import torch
import torchrl.envs as envBase
import copy 
import os.path
import pickle
from utils.tree_utils import *


class BalancingTreeRL(envBase):

    def __init__(self,reuse_per_tree = 10, **kwargs):
        super.__init__()
        self.reuse_per_tree = reuse_per_tree
        self.episode_counter = 0
        self.tree_pyg = bst_to_pyg(self._load_trees("dataset_trees.pkl"))
        self.tree = None
        self.tree_index = 0
        self.og_tree = None
        self.prev_balance = sum(bst_to_pyg(self.tree))
        self.reset()
    
    def _load_trees(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            print(data) 
        return data
        
    def reset(self, **kwargs):
        
        # We setup a new tree and change it every 10 episodes 
        if self.tree is None and self.episode_counter % self.reuse_per_tree != 0 :
            self.tree_index = (self.tree_index + 1) % len(self.tree_pyg)
            self.tree = self.tree[self.tree_index]
            self.og_tree = copy.deep_copy(self.tree)
        
        # We keep using the same tree
        else :
            self.tree = copy.deepcopy(self.og_tree)
        
        return self._get_observations()
    
    def step(self, action):
        self.tree = self.apply_action(self.tree, action)

        reward = self.get_rewards(self.tree)
        obs = self.bst_to_pyg(self.tree)
        done = self.is_done(self.tree)

        return obs, reward, done, {}
    
    def apply_action(self, tree, action):
        if action == 0:
            tree_modified = leftRotate(tree)
        elif action == 1:
            tree_modified = rightRotate(tree)
        else:
            tree_modified = tree
        return tree_modified
    
    def get_rewards(self, root):
        sum = 0
        while root is not None:
            sum += 10 - imbalance 
        return imbalance(tree)
    
    
    def is_done(self, tree):
        return
    
