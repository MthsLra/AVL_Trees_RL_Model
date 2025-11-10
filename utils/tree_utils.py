from binarytree import bst

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