import torch
import torchrl.envs as envBase
import copy 
import os.path
import pickle
from utils.tree_utils import *


data_path = os.path.join("..", "data", "dataset_trees.pkl")


class BalancingTreeRL(envBase):

    def __init__(self,reuse_per_tree = 10, **kwargs):
        super.__init__()

        self.reuse_per_tree = reuse_per_tree
        self.episode_counter = 0
        self.trees = self._load_trees(data_path)
        self.tree_index = -1
        self.tree= None
        self.og_tree = None
        self.prev_imbalance = None 

        self.reset()
    
    def _load_trees(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            print(data) 
        return data
        
    def reset(self, **kwargs):
        
        # We setup a new tree and change it every 10 episodes 
        if self.episode_counter % self.reuse_per_tree == 0 :
            self.tree_index = (self.tree_index + 1) % len(self.tree_pyg)
            self.og_tree = self.trees[self.tree_index]
        
        # We keep using the same tree
        self.tree = copy.deepcopy(self.og_tree)

        # Update the old imbalance 
        self.prev_imbalance = total_imbalance(self.tree)

        self.episode_counter += 1

        obs = bst_to_pyg(self.tree)
        
        return obs
    
    def step(self, action):

        # Apply one of the rotation to the current tree 
        self.tree = self.apply_action(self.tree, action)

        # Compute the new imbalance and the reward is computed as the difference between the imabalance before and after the rotation was applied 
        new_imbalance = total_imbalance(self.tree)
        reward = self.prev_imbalance - new_imbalance
        self.prev_imbalance = new_imbalance

        # The obsesrvation is the rotated tree vectorized as a pyg object 
        obs = self.bst_to_pyg(self.tree)

        # If the rotated tree is an avl tree we finish the episode 
        done = (new_imbalance == 1)

        return obs, reward, done, {}
    
    def apply_action(self, tree, action):
        if action == 0:
            tree_modified = leftRotate(tree)
        elif action == 1:
            tree_modified = rightRotate(tree)
        else:
            tree_modified = tree
        return tree_modified
    
    
