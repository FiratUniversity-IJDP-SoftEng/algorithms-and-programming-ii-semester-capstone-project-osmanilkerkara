import matplotlib.pyplot as plt

def plot_decision_tree(dt, feature_names, class_names):
    """Plot simple decision tree"""
    fig, ax = plt.subplots(figsize=(10, 6))

    def plot_node(node, x, y, dx, dy, depth=0):
        if node is None:
            return

        text = f"{'Class ' + str(node.value) if node.value is not None else f'Feature {node.feature+1} <= {node.threshold:.2f}'}"
        text += f"\nGini: {node.gini:.2f}"

        ax.text(x, y, text, 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                ha='center', va='center')

        if node.left:
            ax.plot([x, x-dx], [y, y-dy], 'k-')
            plot_node(node.left, x-dx, y-dy, dx/2, dy, depth+1)
        if node.right:
            ax.plot([x, x+dx], [y, y-dy], 'k-')
            plot_node(node.right, x+dx, y-dy, dx/2, dy, depth+1)

    plot_node(dt.root, 0, 0, 0.4, -0.3)
    ax.axis('off')
    plt.tight_layout()
    return fig