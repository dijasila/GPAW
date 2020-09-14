import numpy as np


class PointGroup:
    """
    Upper class for point groups that includes general routines
    such as rotations and mirroring operations.
    """
    def __init__(self, name: str):
        import gpaw.point_groups.groups as groups
        self.name = name
        group = getattr(groups, name)()
        self.character_table = np.array(group.character_table)
        self.operations = {}
        for opname, op in group.operations:
            if not isinstance(op, np.ndarray):
                op = op(np.eye(3))
            self.operations[opname] = op
        self.symmetries = group.symmetries
        self.nops = group.nof_operations
        self.complex = getattr(group, 'complex', False)
        self.translations = (group.Tx_i, group.Ty_i, group.Tz_i)

    def __str__(self) -> str:
        lines = [[self.name] + list(self.operations)]
        for sym, coefs in zip(self.symmetries, self.character_table):
            lines.append([sym] + list(coefs))
        return '\n'.join(f'{line[0]:5}' +
                         ''.join(f'{word:>10}' for word in line[1:])
                         for line in lines) + '\n'

    def get_normalized_table(self):
        self.D = [row[0] for row in self.character_table]  # degeneracies
        self.normalized_table = list(map(lambda x, y: list(np.array(x) / y),
                                     self.character_table, self.D))
        return self.normalized_table
