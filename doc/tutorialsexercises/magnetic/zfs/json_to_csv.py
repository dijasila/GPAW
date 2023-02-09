# web-page: zfs_nv.csv
import json
import numpy as np
import csv
from gpaw.zero_field_splitting import convert_tensor


def get_zfs_from_json(name):
    with open(name + '.json', 'r') as fd:
        data = json.load(fd)

    unrelaxed_zfs = np.array(data['D1'])
    relaxed_zfs = np.array(data['D2'])

    return unrelaxed_zfs, relaxed_zfs


def make_list_for_csv(names, D_vvs, unit):
    output = []
    for i in range(len(D_vvs)):
        D, E, axis, scaled_tensor = convert_tensor(D_vvs[i], unit=unit)
        output.append(names[i] + ["{:.0f}".format(D), "{:.2e}".format(E)])

    return output


NC62_unrelaxed_zfs, NC62_relaxed_zfs = get_zfs_from_json('NC62')
NC214_unrelaxed_zfs, NC214_relaxed_zfs = get_zfs_from_json('NC214')

unit = 'MHz'

names = [[62, False], [62, True], [214, False], [214, True]]
D_vvs = [NC62_unrelaxed_zfs,
         NC62_relaxed_zfs,
         NC214_unrelaxed_zfs,
         NC214_relaxed_zfs]

D_E_list = make_list_for_csv(names, D_vvs, unit)

with open('zfs_nv.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(len(D_E_list)):
        writer.writerow(D_E_list[i])
