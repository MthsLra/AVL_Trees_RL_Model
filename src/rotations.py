from binarytree import bst

def rightRotate(y):
  print('Rotate right on node',y.value)
  x = y.left
  T2 = x.right
  x.right = y
  y.left = T2
  return x

def leftRotate(x):
  print('Rotate left on node',x.value)
  y = x.right
  T2 = y.left
  y.left = x
  x.right = T2
  return y
