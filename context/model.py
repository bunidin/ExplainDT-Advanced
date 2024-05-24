from json import load
from typing import Optional


class Node:

    def __init__(self, id: int, parent: Optional['Node'] = None):
        self.id = id
        self.parent = parent

    def get_sub_tree(self) -> list['Node']:
        return [self]

    def count_sub_tree(self) -> int:
        return 1

    def get_sub_tree_depth(self) -> int:
        return 0

    @property
    def is_leaf(self):
        return False

    @property
    def is_poss_leaf(self):
        return False

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)


class FeatNode(Node):

    def __init__(
        self,
        id: int,
        label: int,
        child_zero: Node,
        child_one: Node,
        parent: Optional['FeatNode'] = None
    ):
        super().__init__(id)
        self.label = label
        self.child_zero = child_zero
        self.child_one = child_one
        self.parent = parent

    def get_sub_tree(self) -> list[Node]:
        sub_tree: list['Node'] = [self]
        sub_tree.extend(self.child_zero.get_sub_tree())
        sub_tree.extend(self.child_one.get_sub_tree())
        return sub_tree

    def count_sub_tree(self) -> int:
        count = (
            1 +
            self.child_zero.count_sub_tree() +
            self.child_one.count_sub_tree()
        )
        return count

    def get_sub_tree_depth(self) -> int:
        sub_tree_depth = max(
            self.child_zero.get_sub_tree_depth(),
            self.child_one.get_sub_tree_depth()
        )
        return sub_tree_depth + 1


class Leaf(Node):

    def __init__(
        self,
        id: int,
        truth_value: bool,
        parent: Optional[FeatNode] = None
    ):
        super().__init__(id)
        self.truth_value = truth_value
        self.parent = parent

    @property
    def is_leaf(self):
        return True

    @property
    def is_poss_leaf(self):
        return self.truth_value


class Tree:

    def __init__(
        self,
        root: Optional[FeatNode] = None,
        from_file: Optional[str] = None,
        iterative: Optional[bool] = False
    ):
        self.pos_leafs = []
        self.neg_leafs = []
        self.classes = []
        self.features = []
        if root is not None and from_file is None:
            self.root = root
            return
        if root is None and from_file is not None:
            self.load_tree(from_file, iterative)
            return
        raise Exception('One and only one argument must be given.')

    def depth(self) -> int:
        return self.root.get_sub_tree_depth()

    def number_of_nodes(self) -> int:
        return self.root.count_sub_tree()

    def nodes(self) -> list[Node]:
        return self.root.get_sub_tree()

    def r_generate_nodes(
        self,
        id: int,
        node_dict: dict,
        truth_class: str
    ) -> Node:
        node_info = node_dict[str(id)]
        if node_info['type'] == 'leaf':
            truth_value = node_info['class'] == truth_class
            leaf = Leaf(
                id=node_info['id'],
                truth_value=truth_value
            )
            if truth_value:
                self.pos_leafs.append(leaf)
            else:
                self.neg_leafs.append(leaf)
            return leaf
        node = FeatNode(
            id=node_info['id'],
            label=node_info['feature_index'],
            child_zero=self.r_generate_nodes(
                node_info['id_left'],
                node_dict,
                truth_class
            ),
            child_one=self.r_generate_nodes(
                node_info['id_right'],
                node_dict,
                truth_class
            )
        )
        node.child_zero.parent = node
        node.child_one.parent = node
        return node

    def i_generate_nodes(self, node_dict: dict, truth_class: str) -> None:
        created_node_dict: dict[int, Node] = {}
        # Lets make use of the fact that children always have a higher id than
        # their parents to only iterate once over every node.
        max_node_id = max(map(lambda x: int(x), node_dict.keys()))
        for i in range(max_node_id, -1, -1):
            node_info = node_dict[str(i)]
            if node_info['type'] == 'leaf':
                truth_value = node_info['class'] == truth_class
                created_node_dict[i] = Leaf(
                    id=node_info['id'],
                    truth_value=truth_value
                )
                if truth_value:
                    self.pos_leafs.append(created_node_dict[i])
                else:
                    self.neg_leafs.append(created_node_dict[i])
                continue
            node = FeatNode(
                id=node_info['id'],
                label=node_info['feature_index'],
                child_zero=created_node_dict[node_info['id_left']],
                child_one=created_node_dict[node_info['id_right']],
            )
            created_node_dict[i] = node
            created_node_dict[node_info['id_left']].parent = node
            created_node_dict[node_info['id_right']].parent = node
        self.root = created_node_dict[0]

    def load_tree(
        self,
        file_name: str,
        iterative: Optional[bool] = False
    ) -> None:
        # Load dtree as a dictionary
        file = open(file_name)
        tree_json = load(file)
        file.close()
        # Create tree
        truth_class = tree_json['class_names'][0]
        self.classes = tree_json['class_names']
        self.features = tree_json['feature_names']
        if iterative:
            self.i_generate_nodes(tree_json['nodes'], truth_class)
            return
        self.root = self.r_generate_nodes(0, tree_json['nodes'], truth_class)
