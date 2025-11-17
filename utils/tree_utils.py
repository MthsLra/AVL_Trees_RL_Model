from binarytree import bst
from torch_geometric import data
import torch 
from collections import deque

# Balance analysis 
def imbalance(root):

    if root.left == None and root.right == None:
        return 0
    elif root.left == None:
        return root.right.height
    elif root.right == None:
        return root.left.height
    else:
        return abs(root.left.height - root.right.height)
    
def total_imbalance(root):
    visited = []
    avl_violations = 0
    q = deque()
    q.append(root)
    while not q.empty():
        node = q.pop()
        visited.append(node)
        if abs(imbalance(node)) > 1:
            avl_violations += 1

        if node.left != None and node.left not in visited:
           q.append(node.left)

        if node.right != None and node.right not in visited:
           q.append(node.right)

    return avl_violations
        
    
        
# Rotations functions
def rightRotate(y):
  x = y.left
  T2 = x.right
  x.right = y
  y.left = T2
  return x

def leftRotate(x):
  y = x.right
  T2 = y.left
  y.left = x
  x.right = T2
  return y

# Transform the bst tree as a pyg object (necessary for the tree to be inputed in the neural network)
def bst_to_pyg(root):
   
    if root is None:
        return None
    
    # Traverse nodes and assign ids
    nodes = []
    def dfs(node):
        if node is None:
          return None
        idx = len(nodes)
        nodes.append(node)
        dfs(node.left)
        dfs(node.right)
    dfs(root)

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    feats = []
    for node in nodes:
        bf = imbalance(node)
        feats.append([node.value, bf])

    # Index the edges (directed)
    edge_index = []
    for parent in nodes:
        parent_idx = node_to_idx[parent]
        if parent.left is not None:
          edge_index.append([parent_idx, node_to_idx[parent.left]])
        if parent.right is not None:
          edge_index.append([parent_idx, node_to_idx[parent.right]])

    
    if edge_index:
       edge_index = torch.tensor(edge_index, dtype =torch.long).t().contiguous()

    else:
       edge_index = torch.tensor((2, 0), dtype=torch.long)

    x = torch.tensor(feats, dtype = torch.float)

    return data(x=x, edge_index = edge_index)

