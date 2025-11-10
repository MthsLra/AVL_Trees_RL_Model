from binarytree import bst
import random
import pandas as pd
from collections import deque
import os.path
import pickle 
from utils.tree_utils import imbalance 

def create_dataset():
    trees = {}
    trees_balances = {}
    balances = []
    for i in range(100):
        new_bst = bst(height = random.randint(3, 7))
        trees[i] = list(new_bst.inorder)
        for n in new_bst.inorder:
            balances.append(imbalance(n))
        trees_balances[i] = balances
        balances = []
    return (trees_balances, trees)
    



def main():
    # Path setup
    data_path_trees = os.path.join("..", "data", "dataset_trees.pkl")
    data_path_bal = os.path.join("..", "data", "dataset_balances.pkl")


    # Generate dataset
    (dataset_bal, dataset_trees) = create_dataset()

    
    # Write to file
    with open(data_path_trees, "wb") as f:
        pickle.dump(dataset_trees, f)
        

    with open(data_path_bal, "wb") as f:
        pickle.dump(dataset_bal, f)
    print(f"✅ Dataset successfully written to {data_path_trees} and {data_path_bal}")


if __name__ == "__main__":
    main()
