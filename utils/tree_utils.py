from binarytree import bst
from torch_geometric import data
import torch 

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

# Vectorized form of the tree in terms of how each node is balanced
def vectorize_tree(tree):
    vectorized = []
    for n in tree.inorder:
      vectorized.append(imbalance(n))
    return vectorized 

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

