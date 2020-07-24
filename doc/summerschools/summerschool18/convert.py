# creates: intro/intro.ipynb
# creates: batteries/batteries1.ipynb
# creates: batteries/batteries2.ipynb
# creates: batteries/batteries3.ipynb
# creates: catalysis/n2_on_metal.ipynb, catalysis/neb.ipynb
# creates: catalysis/vibrations.ipynb, catalysis/convergence.ipynb
# creates: magnetism/magnetism1.ipynb, magnetism/magnetism2.ipynb
# creates: magnetism/magnetism3.ipynb,
# creates: machinelearning/machinelearning.ipynb
# creates: photovoltaics/pv1.ipynb, photovoltaics/pv2.ipynb
# creates: photovoltaics/pv3.ipynb
from pathlib import Path
from gpaw.utilities.nbrun import py2ipynb


for path in Path().glob('*/*.py'):
    if path.read_text().startswith('# %%\n'):
        print(path)
        py2ipynb(path)
