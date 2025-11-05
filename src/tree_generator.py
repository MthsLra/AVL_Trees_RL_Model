from binarytree import bst
import random
import pandas as pd
from collections import deque
import os.path
import pickle 

'''
# Implement bfs to make rotations easier
def bfs(root):

    # If tree is empty
    if not root:
        return []
    
    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(node.value)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    
    # Optional: remove trailing None values
    while result and result[-1] is None:
        result.pop()
    
    return result
'''
'''
# Create random bst and put the in an array after applying bfs 
def create_dataset():
    trees = []
    for _ in range(100):
        new_bst = bst(height = random.randint(3, 7))
        trees.append(new_bst.inorder)
    return trees
'''

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

def imbalance(root):

    # If it is a leaf 
    if root.left == None and root.right == None:
        return 0
    elif root.left == None:
        return root.right.height
    elif root.right == None:
        return root.left.height
    else:
        return abs(root.left.height - root.right.height)
    



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
