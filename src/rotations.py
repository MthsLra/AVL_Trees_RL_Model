from binarytree import bst

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
