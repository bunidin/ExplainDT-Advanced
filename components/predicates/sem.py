from components.base import FALSE, TRUE
from components.cnf import CNF
from components.predicates.base import UnaryPredicate
from components.instances import Var, Constant
from context import Context, Symbol, FeatNode, Leaf


class RawRFS(UnaryPredicate):

    @staticmethod
    def _get_node_path_feats(origin: Leaf, context: Context) -> list[Symbol]:
        feats: list[Symbol] = list(Symbol.BOT for _ in range(context.DIM))
        node = origin
        parent: FeatNode | None = node.parent
        while parent is not None:
            if parent.child_one.id == node.id:
                feats[parent.label] = Symbol.ONE
            else:
                feats[parent.label] = Symbol.ZERO
            node = parent
            parent = node.parent
        return feats

    @staticmethod
    def _get_leaf_path(
        node: Leaf,
        cache: dict[int, list[Symbol] | None],
        context: Context
    ) -> list[Symbol]:
        if cache[node.id] is None:
            feats = RawRFS._get_node_path_feats(node, context)
            cache[node.id] = feats
        return cache[node.id]  # type: ignore

    @staticmethod
    def _add_not_a_witness_clauses(
        var: Var,
        node_1: Leaf,
        node_2: Leaf,
        cached_feats: dict[int, list[Symbol] | None],
        context: Context
    ) -> list[int]:
        feats_1 = RawRFS._get_leaf_path(node_1, cached_feats, context)
        feats_2 = RawRFS._get_leaf_path(node_2, cached_feats, context)
        clauses = []
        for i in range(context.DIM):
            if (
                feats_1[i] != Symbol.BOT and
                feats_2[i] != Symbol.BOT and
                feats_1[i] != feats_2[i]
            ):
                clauses.append(-context.V[(var.name, i, Symbol.BOT)])
        return clauses

    @staticmethod
    def _compare_constant_to_leafs(
        constant: Constant,
        leaf_1: Leaf,
        leaf_2: Leaf,
        cached_feats: dict[int, list[Symbol] | None],
        context: Context
    ) -> bool:
        feats_1 = RawRFS._get_leaf_path(leaf_1, cached_feats, context)
        feats_2 = RawRFS._get_leaf_path(leaf_2, cached_feats, context)
        for i, feat in enumerate(constant.value):
            consistent_feat = (
                feat != Symbol.BOT and
                feats_1[i] != Symbol.BOT and
                feats_2[i] != Symbol.BOT and
                feats_1[i] != feats_2[i]
            )
            if consistent_feat:
                return True
        return False

    @staticmethod
    def _var_encode(var: Var, context: Context) -> CNF:
        cnf = CNF()
        leafs = context.TREE.pos_leafs + context.TREE.neg_leafs
        cached_feats: dict[int, list[Symbol] | None] = {
            leaf.id: None
            for leaf in leafs
        }
        for p_leaf in context.TREE.pos_leafs:
            for n_leaf in context.TREE.neg_leafs:
                cnf.append(RawRFS._add_not_a_witness_clauses(
                    var,
                    p_leaf,
                    n_leaf,
                    cached_feats,
                    context
                ))
        return cnf

    @staticmethod
    def _constant_encode(constant: Constant, context: Context) -> CNF:
        leafs = context.TREE.pos_leafs + context.TREE.neg_leafs
        cached_feats: dict[int, list[Symbol] | None] = {
            leaf.id: None
            for leaf in leafs
        }
        for p_leaf in context.TREE.pos_leafs:
            for n_leaf in context.TREE.neg_leafs:
                if not RawRFS._compare_constant_to_leafs(
                    constant,
                    p_leaf,
                    n_leaf,
                    cached_feats,
                    context
                ):
                    return CNF(from_clauses=[[]])
        return CNF()

    def simplify(self):
        if isinstance(self.instance, Var):
            return self
        leafs = self.context.TREE.pos_leafs + self.context.TREE.neg_leafs
        cached_feats: dict[int, list[Symbol] | None] = {
            leaf.id: None
            for leaf in leafs
        }
        for p_leaf in self.context.TREE.pos_leafs:
            for n_leaf in self.context.TREE.neg_leafs:
                if not RawRFS._compare_constant_to_leafs(
                    self.instance,
                    p_leaf,
                    n_leaf,
                    cached_feats,
                    self.context
                ):
                    return FALSE()
        return TRUE()


class NotRFS(UnaryPredicate):

    @staticmethod
    def _add_variables(var: Var, context: Context) -> None:
        # Feature variables
        for i in range(context.DIM):
            context.add_var(('s1', var.name, i, Symbol.ONE))
            context.add_var(('s2', var.name, i, Symbol.ONE))
            # Sameness variables
            context.add_var(('neq1', var.name, i))
            context.add_var(('neq2', var.name, i))
        # Reachability variables
        for node in context.TREE.nodes():
            context.add_var(('r', 's1', var.name, node.id))
            context.add_var(('r', 's2', var.name, node.id))

    @staticmethod
    def _create_var_evaluation(
        var_name: tuple[str, str],
        truth_value: bool,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        clauses.append(
            [
                context.V[(
                    'r',
                    *var_name,
                    context.TREE.root.id,
                )]
            ],
        )
        # Add propagation clauses
        for node in context.TREE.nodes():
            if node.is_leaf and node.is_poss_leaf != truth_value:
                clauses.append([-context.V[('r', *var_name, node.id)]])
            elif isinstance(node, FeatNode):
                # r_x_i ^ x_i = 0 -> r_x_i.zero
                clauses.append(
                    [
                        context.V[(*var_name, node.label, Symbol.ONE)],
                        -context.V[('r', *var_name, node.id)],
                        context.V[('r', *var_name, node.child_zero.id)]
                    ],
                )
                # r_x_i ^ x_i = 1 -> r_x_i.one
                clauses.append(
                    [
                        -context.V[(*var_name, node.label, Symbol.ONE)],
                        -context.V[('r', *var_name, node.id)],
                        context.V[('r', *var_name, node.child_one.id)]
                    ],
                )
                # r_x_i.one -> r_x_i ^ x_i = 1
                clauses.append(
                    [
                        -context.V[('r', *var_name, node.child_one.id)],
                        context.V[('r', *var_name, node.id)]
                    ],
                )
                clauses.append(
                    [
                        -context.V[('r', *var_name, node.child_one.id)],
                        context.V[(*var_name, node.label, Symbol.ONE)]
                    ],
                )
                # r_x_i.zero -> r_x_i ^ x_i = 0
                clauses.append(
                    [
                        -context.V[('r', *var_name, node.child_zero.id)],
                        context.V[('r', *var_name, node.id)]
                    ],
                )
                clauses.append(
                    [
                        -context.V[('r', *var_name, node.child_zero.id)],
                        -context.V[(*var_name, node.label, Symbol.ONE)]
                    ],
                )
        return clauses

    @staticmethod
    def _create_consistencty_clauses(
        var: Var,
        context: Context
    ) -> list[list[int]]:
        clauses = []
        # Vars share values on feats where var is not bot
        for i in range(context.DIM):
            clauses.extend([
                # s1_i=1 ^ s2_i!=1 -> neq1
                [
                    -context.V[('s1', var.name, i, Symbol.ONE)],
                    context.V[('s2', var.name, i, Symbol.ONE)],
                    context.V[('neq1', var.name, i)]
                ],
                # neq1 -> s1_i=1 ^ s2_i!=1
                [
                    -context.V[('neq1', var.name, i)],
                    context.V[('s1', var.name, i, Symbol.ONE)],
                ],
                [
                    -context.V[('neq1', var.name, i)],
                    -context.V[('s2', var.name, i, Symbol.ONE)],
                ],
                # s1_i!=1 ^ s2_i=1 -> neq2
                [
                    context.V[('s1', var.name, i, Symbol.ONE)],
                    -context.V[('s2', var.name, i, Symbol.ONE)],
                    context.V[('neq2', var.name, i)]
                ],
                # neq2 -> s1_i!=1 ^ s2_i=1
                [
                    -context.V[('neq2', var.name, i)],
                    -context.V[('s1', var.name, i, Symbol.ONE)],
                ],
                [
                    -context.V[('neq2', var.name, i)],
                    context.V[('s2', var.name, i, Symbol.ONE)],
                ],
            ])
        for i in range(context.DIM):
            # s1 and s2 can differ only on the features where var has a bot
            clauses.extend([
                [
                    context.V[(var.name, i, Symbol.BOT)],
                    -context.V[('neq1', var.name, i)],
                ],
                [
                    context.V[(var.name, i, Symbol.BOT)],
                    -context.V[('neq2', var.name, i)],
                ],
            ])
        # # Evaluate var
        clauses.extend(
            NotRFS._create_var_evaluation(('s1', var.name), True, context)
        )
        clauses.extend(
            NotRFS._create_var_evaluation(('s2', var.name), False, context)
        )
        return clauses

    @staticmethod
    def _var_encode(var: Var, context: Context) -> CNF:
        cnf = CNF()
        # Create variables
        NotRFS._add_variables(var, context)
        # We encode not(RFS(var))
        cnf.extend(
            NotRFS._create_consistencty_clauses(var, context),
            in_consistency=True
        )
        return cnf

    @staticmethod
    def _constant_encode(constant: Constant, context: Context) -> CNF:
        raise Exception('Not yet implemented')

    def simplify(self):
        return super().simplify()


class RFS(RawRFS):
    pass
