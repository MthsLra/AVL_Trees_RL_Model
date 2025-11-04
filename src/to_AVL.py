from binarytree import Node
from rotations import leftRotate, rightRotate


def get_height(node):
    """Calculate the height of a node."""
    if not node:
        return 0
    return 1 + max(get_height(node.left), get_height(node.right))


def get_balance(node):
    """Calculate the balance factor of a node."""
    if not node:
        return 0
    return get_height(node.left) - get_height(node.right)


def bst_to_avl(node):
    """
    Convert a BST to an AVL tree using rotations.
    
    Args:
        node: Root node of the BST
        
    Returns:
        Root node of the balanced AVL tree
    """
    # Base case: empty node
    if not node:
        return node
    
    # Recursively balance left and right subtrees first
    if node.left:
        node.left = bst_to_avl(node.left)
    if node.right:
        node.right = bst_to_avl(node.right)
    
    # Get the balance factor of this node
    balance = get_balance(node)
    
    # If node is unbalanced, there are 4 cases:
    
    # Left-Left Case
    if balance > 1 and get_balance(node.left) >= 0:
        return rightRotate(node)
    
    # Right-Right Case
    if balance < -1 and get_balance(node.right) <= 0:
        return leftRotate(node)
    
    # Left-Right Case
    if balance > 1 and get_balance(node.left) < 0:
        node.left = leftRotate(node.left)
        return rightRotate(node)
    
    # Right-Left Case
    if balance < -1 and get_balance(node.right) > 0:
        node.right = rightRotate(node.right)
        return leftRotate(node)
    
    # Node is already balanced
    return node


def verify_avl(node):
    """
    Verify if a tree is a valid AVL tree.
    
    Args:
        node: Root node of the tree
        
    Returns:
        True if the tree is a valid AVL tree, False otherwise
    """
    if not node:
        return True
    
    balance = get_balance(node)
    
    # Check if balance factor is within AVL constraints
    if abs(balance) > 1:
        return False
    
    # Recursively check subtrees
    return verify_avl(node.left) and verify_avl(node.right)


# Example usage
if __name__ == "__main__":
    from binarytree import bst
    
    # Create a random BST
    print("Creating a random BST...")
    tree = bst(height=4)
    print("\nOriginal BST:")
    print(tree)
    
    # Convert to AVL
    print("\nConverting to AVL tree...")
    avl_root = bst_to_avl(tree)
    print("\nBalanced AVL tree:")
    print(avl_root)
    
    # Verify it's a valid AVL tree
    is_avl = verify_avl(avl_root)
    print(f"\nIs valid AVL tree? {is_avl}")
    print(f"Max balance factor in tree: {max(abs(get_balance(avl_root)), abs(get_balance(avl_root.left)) if avl_root.left else 0, abs(get_balance(avl_root.right)) if avl_root.right else 0)}")
