from functools import reduce
from copy import deepcopy
from typing import Optional


# CNF implementation HEAVILY inspired by PySat's CNF.


class CNF:

    def __init__(self, from_clauses: Optional[list[list[int]]] = None):
        self.nv = 0
        self.meaning_clauses: list[list[int]] = []
        self.consistency_clauses: list[list[int]] = []
        self.negated: bool = False

        if from_clauses is not None:
            self.from_clauses(from_clauses)

    def from_clauses(self, clauses) -> None:
        self.meaning_clauses = deepcopy(clauses)
        for cl in self.meaning_clauses:
            self.nv = max([abs(lit) for lit in cl] + [self.nv])

    def generate_negation(
        self,
        nv: int,
        clauses: list[list[int]]
    ) -> tuple[list[list[int]], list[int], int]:
        nv = nv
        clauses = []
        enclits = []
        for cl in self.meaning_clauses:
            auxv = -cl[0]
            if len(cl) > 1:
                nv += 1
                auxv = nv
                # direct implication
                for lit in cl:
                    clauses.append([-lit, -auxv])
                # opposite implication
                clauses.append(cl + [auxv])
            # literals representing negated clauses
            enclits.append(auxv)
        return clauses, enclits, nv

    def add_negation_iff_clauses(
        self,
        negated: 'CNF',
        enclits: list[int]
    ) -> None:
        negated.nv += 1
        auxv = negated.nv
        for lit in enclits:
            negated.consistency_clauses.append([-lit, auxv])
        negated.consistency_clauses.append(enclits + [-auxv])
        negated.meaning_clauses.append([auxv])

    def negate(self, topv: Optional[int] = None) -> 'CNF':
        # Handle empty cnf case.
        if len(self.meaning_clauses) == 0:
            return CNF(from_clauses=[[]])
        # Handle empty clause case.
        if len(self.meaning_clauses) > 0 and reduce(
            lambda x, y: x or y,
            map(lambda cl: len(cl) == 0, self.meaning_clauses)
        ):
            return CNF()
        # Create new cnf and transfer consistency clauses.
        negated = CNF()
        negated.consistency_clauses.extend(self.consistency_clauses)
        # If already negated then return simple negation.
        if self.negated:
            assert len(self.meaning_clauses) == 1
            assert len(self.meaning_clauses[0]) == 1
            negated.append([-self.meaning_clauses[0][0]])
            return negated
        # Apply transformation.
        topv = topv if topv is not None else self.nv
        consistency_clauses, enclits, neg_nv = self.generate_negation(
            topv,
            self.meaning_clauses
        )
        negated.nv = neg_nv
        negated.consistency_clauses.extend(consistency_clauses)
        # Generate bidirection implication from new enc literal and enclits.
        if len(enclits) == 1:
            negated.meaning_clauses.append(enclits)
            return negated
        elif len(enclits) > 1:
            self.add_negation_iff_clauses(negated, enclits)
        return negated

    def conjunction(self, cnf: 'CNF') -> 'CNF':
        cnf_conjunction = CNF()
        cnf_conjunction.meaning_clauses.extend(self.meaning_clauses)
        cnf_conjunction.meaning_clauses.extend(cnf.meaning_clauses)
        cnf_conjunction.consistency_clauses.extend(self.consistency_clauses)
        cnf_conjunction.consistency_clauses.extend(cnf.consistency_clauses)
        cnf_conjunction.nv = max(self.nv, cnf.nv)
        return cnf_conjunction

    def _append(
        self,
        clause: list[int],
        in_consistency: bool
    ) -> None:
        self.nv = max([abs(lit) for lit in clause] + [self.nv])
        if in_consistency:
            self.consistency_clauses.append(clause)
            return
        self.meaning_clauses.append(clause)

    def append(
        self,
        clause: list[int],
        in_consistency=False
    ) -> None:
        self.negated = False
        self._append(clause, in_consistency)

    def extend(
        self,
        clauses: list[list[int]],
        in_consistency=False
    ) -> None:
        self.negated = False
        for cl in clauses:
            self._append(cl, in_consistency)

    def to_file(
        self,
        file_name: str
    ) -> None:
        clauses = self.meaning_clauses + self.consistency_clauses
        with open(file_name, 'w') as file:
            print('p cnf', self.nv, len(clauses), file=file)
            for clause in clauses:
                print(' '.join(str(i) for i in clause), '0', file=file)
