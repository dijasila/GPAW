import numpy as np


class PointGroup:
    """
    Upper class for point groups that includes general routines
    such as rotations and mirroring operations.
    """
    def __init__(self, name):
        import gpaw.point_groups.groups as groups
        group = getattr(groups, name)()
        self.character_table = np.array(group.character_table)
        self.operations = {}
        for opname, op in group.operations:
            if not isinstance(op, np.ndarray):
                op = op(np.eye(3))
            self.operations[opname] = op
        self.symmetries = group.symmetries

    def __str__(self) -> str:
        name = self.__class__.__name__
        lines = [[name] + self.opnames]
        for sym, coefs in zip(self.symnames, self.character_table):
            lines.append([sym] + list(coefs))
        return '\n'.join(f'{line[0]}' +
                         ''.join(f'{word:>8}' for word in line[1:])
                         for line in lines) + '\n'

    def make_reducable_table(self):
        """
        Create a reducable table out of the irreducable one.
        Create also symmetry representations for translation operators
        Tx,Ty,Tz.
        """

        reducable_character_table = []
        for i, row in enumerate(self.character_table):
            reducable_character_table.append([])
            for j, num in enumerate(row):
                for k in range(self.nof_operations[j]):
                    reducable_character_table[-1].append(num)
        self.reducable_character_table = reducable_character_table
        self.Tx = self.reducable_character_table[self.Tx_i]
        self.Ty = self.reducable_character_table[self.Ty_i]
        self.Tz = self.reducable_character_table[self.Tz_i]
        return 1

    def get_normalized_table(self):
        self.D = [row[0] for row in self.character_table]  # degeneracies
        self.normalized_table = list(map(lambda x, y: list(np.array(x) / y),
                                     self.character_table, self.D))
        return self.normalized_table
