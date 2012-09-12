import os

import sys

import glob

from ase.test.tasks.analyse import AnalyseSCFTask as Task

rundefs = {
    #
    'm103poisson': 'Mixer(0.10, 3)',
    #
    'm103mp': 'Mixer(0.10, 3)',
    'd203mp': 'MixerDif(0.20, 3)',
    'dzpd203mp': 'MixerDif(0.20, 3)',
    'cgm203mp': 'MixerDif(0.20, 3)',
    'cgdzpm203mp': 'MixerDif(0.20, 3)',
    'cgdzpd203mp': 'MixerDif(0.20, 3)',
    #
    'm251': 'Mixer(0.25, 1)',
    'm252': 'Mixer(0.25, 2)',
    'm253': 'Mixer(0.25, 3)',
    'm254': 'Mixer(0.25, 4)',
    'm255': 'Mixer(0.25, 5)',
    'm256': 'Mixer(0.25, 6)',
    'm257': 'Mixer(0.25, 7)',
    #
    'm201': 'Mixer(0.20, 1)',
    'm202': 'Mixer(0.20, 2)',
    'm203': 'Mixer(0.20, 3)',
    'm204': 'Mixer(0.20, 4)',
    'm205': 'Mixer(0.20, 5)',
    'm206': 'Mixer(0.20, 6)',
    'm207': 'Mixer(0.20, 7)',
    #
    'm151': 'Mixer(0.15, 1)',
    'm152': 'Mixer(0.15, 2)',
    'm153': 'Mixer(0.15, 3)',
    'm154': 'Mixer(0.15, 4)',
    'm155': 'Mixer(0.15, 5)',
    'm156': 'Mixer(0.15, 6)',
    'm157': 'Mixer(0.15, 7)',
    #
    'inititer00m': 'Mixer()',
    'inititer01m': 'Mixer()',
    'inititer02m': 'Mixer()',
    'inititer03m': 'Mixer()',
    'inititer04m': 'Mixer()',
    'inititer05m': 'Mixer()',
    'inititer06m': 'Mixer()',
    'inititer07m': 'Mixer()',
    'inititer08m': 'Mixer()',
    'inititer09m': 'Mixer()',
    'inititer10m': 'Mixer()',
    'inititer15m': 'Mixer()',
    'inititer20m': 'Mixer()',
    #
    'inititer05m051': 'Mixer(0.05, 1)',
    'inititer05m052': 'Mixer(0.05, 2)',
    'inititer05m101': 'Mixer(0.10, 1)',
    'inititer05m102': 'Mixer(0.10, 2)',
    'inititer10m051': 'Mixer(0.05, 1)',
    'inititer10m052': 'Mixer(0.05, 2)',
    'inititer10m101': 'Mixer(0.10, 1)',
    'inititer10m102': 'Mixer(0.10, 2)',
    'inititer20m051': 'Mixer(0.05, 1)',
    'inititer20m052': 'Mixer(0.05, 2)',
    'inititer20m101': 'Mixer(0.10, 1)',
    'inititer20m102': 'Mixer(0.10, 2)',
    #
    'inititer00d': 'MixerDif()',
    'inititer01d': 'MixerDif()',
    'inititer02d': 'MixerDif()',
    'inititer03d': 'MixerDif()',
    'inititer04d': 'MixerDif()',
    'inititer05d': 'MixerDif()',
    'inititer06d': 'MixerDif()',
    'inititer07d': 'MixerDif()',
    'inititer08d': 'MixerDif()',
    'inititer09d': 'MixerDif()',
    'inititer10d': 'MixerDif()',
    'inititer15d': 'MixerDif()',
    'inititer20d': 'MixerDif()',
    #
    'inititer05d051': 'MixerDif(0.05, 1)',
    'inititer05d052': 'MixerDif(0.05, 2)',
    'inititer05d101': 'MixerDif(0.10, 1)',
    'inititer05d102': 'MixerDif(0.10, 2)',
    'inititer10d051': 'MixerDif(0.05, 1)',
    'inititer10d052': 'MixerDif(0.05, 2)',
    'inititer10d101': 'MixerDif(0.10, 1)',
    'inititer10d102': 'MixerDif(0.10, 2)',
    'inititer20d051': 'MixerDif(0.05, 1)',
    'inititer20d052': 'MixerDif(0.05, 2)',
    'inititer20d101': 'MixerDif(0.10, 1)',
    'inititer20d102': 'MixerDif(0.10, 2)',
    #
    'bands00m': 'Mixer()',
    'bands01m': 'Mixer()',
    'bands02m': 'Mixer()',
    'bands03m': 'Mixer()',
    'bands04m': 'Mixer()',
    'bands05m': 'Mixer()',
    'bands06m': 'Mixer()',
    'bands07m': 'Mixer()',
    'bands08m': 'Mixer()',
    'bands09m': 'Mixer()',
    'bands10m': 'Mixer()',
    'bands15m': 'Mixer()',
    'bands20m': 'Mixer()',
    #
    'bands01cgm': 'Mixer()',
    'bands02cgm': 'Mixer()',
    'bands03cgm': 'Mixer()',
    'bands04cgm': 'Mixer()',
    'bands05cgm': 'Mixer()',
    'bands06cgm': 'Mixer()',
    'bands07cgm': 'Mixer()',
    'bands08cgm': 'Mixer()',
    'bands09cgm': 'Mixer()',
    'bands10cgm': 'Mixer()',
    'bands15cgm': 'Mixer()',
    'bands20cgm': 'Mixer()',
    #
    'mw1': 'Mixer(weight=1)',
    'mw25': 'Mixer(weight=25)',
    'mw50': 'Mixer(weight=50)',
    'mw100': 'Mixer(weight=100)',
    'mw200': 'Mixer(weight=200)',
    #
    'm101': 'Mixer(0.10, 1)',
    'm102': 'Mixer(0.10, 2)',
    'm103': 'Mixer(0.10, 3)', # default
    'm104': 'Mixer(0.10, 4)',
    'm105': 'Mixer(0.10, 5)',
    'm106': 'Mixer(0.10, 6)',
    'm107': 'Mixer(0.10, 7)',
    #
    'm051': 'Mixer(0.05, 1)',
    'm052': 'Mixer(0.05, 2)',
    'm053': 'Mixer(0.05, 3)',
    'm054': 'Mixer(0.05, 4)',
    'm055': 'Mixer(0.05, 5)',
    'm056': 'Mixer(0.05, 6)',
    'm057': 'Mixer(0.05, 7)',
    #
    'm302': 'Mixer(0.30, 2)',
    'm303': 'Mixer(0.30, 3)',
    'm304': 'Mixer(0.30, 4)',
    'm305': 'Mixer(0.30, 5)',
    'm306': 'Mixer(0.30, 6)',
    'm307': 'Mixer(0.30, 7)',
    'm308': 'Mixer(0.30, 8)',
    'm352': 'Mixer(0.35, 2)',
    'm353': 'Mixer(0.35, 3)',
    'm354': 'Mixer(0.35, 4)',
    'm355': 'Mixer(0.35, 5)',
    'm356': 'Mixer(0.35, 6)',
    'm357': 'Mixer(0.35, 7)',
    'm358': 'Mixer(0.35, 8)',
    'm402': 'Mixer(0.40, 2)',
    'm403': 'Mixer(0.40, 3)',
    'm404': 'Mixer(0.40, 4)',
    'm405': 'Mixer(0.40, 5)',
    'm406': 'Mixer(0.40, 6)',
    'm407': 'Mixer(0.40, 7)',
    'm408': 'Mixer(0.40, 8)',
    #
    's102': 'MixerSum(0.10, 2)',
    's103': 'MixerSum(0.10, 3)',
    's104': 'MixerSum(0.10, 4)',
    's105': 'MixerSum(0.10, 5)',
    's106': 'MixerSum(0.10, 6)',
    's107': 'MixerSum(0.10, 7)',
    's203': 'MixerSum(0.20, 3)',
    's253': 'MixerSum(0.25, 3)',
    #
    'dzps102': 'MixerSum(0.10, 2)',
    'dzps103': 'MixerSum(0.10, 3)',
    'dzps104': 'MixerSum(0.10, 4)',
    'dzps105': 'MixerSum(0.10, 5)',
    'dzps106': 'MixerSum(0.10, 6)',
    'dzps107': 'MixerSum(0.10, 7)',
    'dzps203': 'MixerSum(0.20, 3)',
    'dzps253': 'MixerSum(0.25, 3)',
    #
    'cgdzps102': 'MixerSum(0.10, 2)',
    'cgdzps103': 'MixerSum(0.10, 3)',
    'cgdzps104': 'MixerSum(0.10, 4)',
    'cgdzps105': 'MixerSum(0.10, 5)',
    'cgdzps106': 'MixerSum(0.10, 6)',
    'cgdzps107': 'MixerSum(0.10, 7)',
    'cgdzps203': 'MixerSum(0.20, 3)',
    'cgdzps253': 'MixerSum(0.25, 3)',
    #
    'b103': 'BroydenMixer(0.10, 3)',
    'b104': 'BroydenMixer(0.10, 4)',
    'b105': 'BroydenMixer(0.10, 5)',
    'b106': 'BroydenMixer(0.10, 6)',
    'b107': 'BroydenMixer(0.10, 7)',
    'b203': 'BroydenMixer(0.20, 3)',
    'b253': 'BroydenMixer(0.25, 3)',
    'b206': 'BroydenMixer(0.20, 6)',
    'b256': 'BroydenMixer(0.25, 6)',
    #
    'cgb103': 'BroydenMixer(0.10, 3)',
    'cgb104': 'BroydenMixer(0.10, 4)',
    'cgb105': 'BroydenMixer(0.10, 5)',
    'cgb106': 'BroydenMixer(0.10, 6)',
    'cgb107': 'BroydenMixer(0.10, 7)',
    #
    'cgdzpb103': 'BroydenMixer(0.10, 3)',
    'cgdzpb104': 'BroydenMixer(0.10, 4)',
    'cgdzpb105': 'BroydenMixer(0.10, 5)',
    'cgdzpb106': 'BroydenMixer(0.10, 6)',
    'cgdzpb107': 'BroydenMixer(0.10, 7)',
    'cgdzpb203': 'BroydenMixer(0.20, 3)',
    'cgdzpb206': 'BroydenMixer(0.20, 6)',
    #
    'dw1': 'MixerDif(weight=1)',
    'dw25': 'MixerDif(weight=25)',
    'dw50': 'MixerDif(weight=50)',
    'dw100': 'MixerDif(weight=100)',
    'd101': 'MixerDif(0.10, 1)',
    'd102': 'MixerDif(0.10, 2)',
    'd103': 'MixerDif(0.10, 3)',
    'd104': 'MixerDif(0.10, 4)',
    'd105': 'MixerDif(0.10, 5)',
    'd106': 'MixerDif(0.10, 6)',
    'd107': 'MixerDif(0.10, 7)',
    'd108': 'MixerDif(0.10, 8)',
    'd152': 'MixerDif(0.15, 2)',
    'd153': 'MixerDif(0.15, 3)',
    'd154': 'MixerDif(0.15, 4)',
    'd155': 'MixerDif(0.15, 5)',
    'd156': 'MixerDif(0.15, 6)',
    'd157': 'MixerDif(0.15, 7)',
    'd158': 'MixerDif(0.15, 8)',
    'd202': 'MixerDif(0.20, 2)',
    'd203': 'MixerDif(0.20, 3)',
    'd204': 'MixerDif(0.20, 4)',
    'd205': 'MixerDif(0.20, 5)',
    'd206': 'MixerDif(0.20, 6)',
    'd207': 'MixerDif(0.20, 7)',
    'd208': 'MixerDif(0.20, 8)',
    'd252': 'MixerDif(0.25, 2)',
    'd253': 'MixerDif(0.25, 3)',
    'd254': 'MixerDif(0.25, 4)',
    'd255': 'MixerDif(0.25, 5)',
    'd256': 'MixerDif(0.25, 6)',
    'd257': 'MixerDif(0.25, 7)',
    'd258': 'MixerDif(0.25, 8)',
    'd302': 'MixerDif(0.30, 2)',
    'd303': 'MixerDif(0.30, 3)',
    'd304': 'MixerDif(0.30, 4)',
    'd305': 'MixerDif(0.30, 5)',
    'd306': 'MixerDif(0.30, 6)',
    'd307': 'MixerDif(0.30, 7)',
    'd308': 'MixerDif(0.30, 8)',
    'd352': 'MixerDif(0.35, 2)',
    'd353': 'MixerDif(0.35, 3)',
    'd354': 'MixerDif(0.35, 4)',
    'd355': 'MixerDif(0.35, 5)',
    'd356': 'MixerDif(0.35, 6)',
    'd357': 'MixerDif(0.35, 7)',
    'd358': 'MixerDif(0.35, 8)',
    'd402': 'MixerDif(0.40, 2)',
    'd403': 'MixerDif(0.40, 3)',
    'd404': 'MixerDif(0.40, 4)',
    'd405': 'MixerDif(0.40, 5)',
    'd406': 'MixerDif(0.40, 6)',
    'd407': 'MixerDif(0.40, 7)',
    'd408': 'MixerDif(0.40, 8)',
    #
    'dzpdw1': 'MixerDif(weight=1)',
    'dzpdw25': 'MixerDif(weight=25)',
    'dzpdw50': 'MixerDif(weight=50)',
    'dzpdw100': 'MixerDif(weight=100)',
    'dzpd101': 'MixerDif(0.10, 1)',
    'dzpd102': 'MixerDif(0.10, 2)',
    'dzpd103': 'MixerDif(0.10, 3)',
    'dzpd104': 'MixerDif(0.10, 4)',
    'dzpd105': 'MixerDif(0.10, 5)',
    'dzpd106': 'MixerDif(0.10, 6)',
    'dzpd107': 'MixerDif(0.10, 7)',
    'dzpd152': 'MixerDif(0.15, 2)',
    'dzpd153': 'MixerDif(0.15, 3)',
    'dzpd154': 'MixerDif(0.15, 4)',
    'dzpd155': 'MixerDif(0.15, 5)',
    'dzpd156': 'MixerDif(0.15, 6)',
    'dzpd157': 'MixerDif(0.15, 7)',
    'dzpd202': 'MixerDif(0.20, 2)',
    'dzpd203': 'MixerDif(0.20, 3)',
    'dzpd204': 'MixerDif(0.20, 4)',
    'dzpd205': 'MixerDif(0.20, 5)',
    'dzpd206': 'MixerDif(0.20, 6)',
    'dzpd207': 'MixerDif(0.20, 7)',
    'dzpd252': 'MixerDif(0.25, 2)',
    'dzpd253': 'MixerDif(0.25, 3)',
    'dzpd254': 'MixerDif(0.25, 4)',
    'dzpd255': 'MixerDif(0.25, 5)',
    'dzpd256': 'MixerDif(0.25, 6)',
    'dzpd257': 'MixerDif(0.25, 7)',
    #
    'szdzpm': 'Mixer()',
    'szpdzpm': 'Mixer()',
    #
    'dzpmw1': 'Mixer(weight=1)',
    'dzpmw25': 'Mixer(weight=25)',
    'dzpmw50': 'Mixer(weight=50)',
    'dzpmw100': 'Mixer(weight=100)',
    'dzpm102': 'Mixer(0.10, 2)',
    'dzpm103': 'Mixer(0.10, 3)',
    'dzpm104': 'Mixer(0.10, 4)',
    'dzpm105': 'Mixer(0.10, 5)',
    'dzpm106': 'Mixer(0.10, 6)',
    'dzpm107': 'Mixer(0.10, 7)',
    'dzpm152': 'Mixer(0.15, 2)',
    'dzpm153': 'Mixer(0.15, 3)',
    'dzpm154': 'Mixer(0.15, 4)',
    'dzpm155': 'Mixer(0.15, 5)',
    'dzpm202': 'Mixer(0.20, 2)',
    'dzpm203': 'Mixer(0.20, 3)',
    'dzpm204': 'Mixer(0.20, 4)',
    'dzpm205': 'Mixer(0.20, 5)',
    'dzpm252': 'Mixer(0.25, 2)',
    'dzpm253': 'Mixer(0.25, 3)',
    'dzpm254': 'Mixer(0.25, 4)',
    'dzpm255': 'Mixer(0.25, 5)',
    #
    'dzpbands00m': 'Mixer()',
    'dzpbands01m': 'Mixer()',
    'dzpbands02m': 'Mixer()',
    'dzpbands03m': 'Mixer()',
    'dzpbands04m': 'Mixer()',
    'dzpbands05m': 'Mixer()',
    'dzpbands06m': 'Mixer()',
    'dzpbands07m': 'Mixer()',
    'dzpbands08m': 'Mixer()',
    'dzpbands09m': 'Mixer()',
    'dzpbands10m': 'Mixer()',
    'dzpbands15m': 'Mixer()',
    'dzpbands20m': 'Mixer()',
    #
    'dzpbands00d': 'MixerDiff()',
    'dzpbands01d': 'MixerDiff()',
    'dzpbands02d': 'MixerDiff()',
    'dzpbands03d': 'MixerDiff()',
    'dzpbands04d': 'MixerDiff()',
    'dzpbands05d': 'MixerDiff()',
    'dzpbands06d': 'MixerDiff()',
    'dzpbands07d': 'MixerDiff()',
    'dzpbands08d': 'MixerDiff()',
    'dzpbands09d': 'MixerDiff()',
    'dzpbands10d': 'MixerDiff()',
    'dzpbands15d': 'MixerDiff()',
    'dzpbands20d': 'MixerDiff()',
    #
    'cgm101': 'Mixer(0.10, 1)',
    'cgm102': 'Mixer(0.10, 2)',
    'cgm103': 'Mixer(0.10, 3)',
    'cgm104': 'Mixer(0.10, 4)',
    'cgm105': 'Mixer(0.10, 5)',
    'cgm106': 'Mixer(0.10, 6)',
    'cgm107': 'Mixer(0.10, 7)',
    'cgm152': 'Mixer(0.15, 2)',
    'cgm153': 'Mixer(0.15, 3)',
    'cgm154': 'Mixer(0.15, 4)',
    'cgm155': 'Mixer(0.15, 5)',
    'cgm156': 'Mixer(0.15, 6)',
    'cgm157': 'Mixer(0.15, 7)',
    'cgm201': 'Mixer(0.20, 2)',
    'cgm202': 'Mixer(0.20, 2)',
    'cgm203': 'Mixer(0.20, 3)',
    'cgm204': 'Mixer(0.20, 4)',
    'cgm205': 'Mixer(0.20, 5)',
    'cgm206': 'Mixer(0.20, 6)',
    'cgm207': 'Mixer(0.20, 7)',
    'cgm252': 'Mixer(0.25, 2)',
    'cgm253': 'Mixer(0.25, 3)',
    'cgm254': 'Mixer(0.25, 4)',
    'cgm255': 'Mixer(0.25, 5)',
    'cgm256': 'Mixer(0.25, 6)',
    'cgm257': 'Mixer(0.25, 7)',
    #
    'cgbands00m': 'Mixer()',
    'cgbands01m': 'Mixer()',
    'cgbands02m': 'Mixer()',
    'cgbands03m': 'Mixer()',
    'cgbands04m': 'Mixer()',
    'cgbands05m': 'Mixer()',
    'cgbands06m': 'Mixer()',
    'cgbands07m': 'Mixer()',
    'cgbands08m': 'Mixer()',
    'cgbands09m': 'Mixer()',
    'cgbands10m': 'Mixer()',
    'cgbands15m': 'Mixer()',
    'cgbands20m': 'Mixer()',
    #
    'cgdzpm102': 'Mixer(0.10, 2)',
    'cgdzpm103': 'Mixer(0.10, 3)',
    'cgdzpm104': 'Mixer(0.10, 4)',
    'cgdzpm105': 'Mixer(0.10, 5)',
    'cgdzpm152': 'Mixer(0.15, 2)',
    'cgdzpm153': 'Mixer(0.15, 3)',
    'cgdzpm154': 'Mixer(0.15, 4)',
    'cgdzpm155': 'Mixer(0.15, 5)',
    'cgdzpm202': 'Mixer(0.20, 2)',
    'cgdzpm203': 'Mixer(0.20, 3)',
    'cgdzpm204': 'Mixer(0.20, 4)',
    'cgdzpm205': 'Mixer(0.20, 5)',
    'cgdzpm252': 'Mixer(0.25, 2)',
    'cgdzpm253': 'Mixer(0.25, 3)',
    'cgdzpm254': 'Mixer(0.25, 4)',
    'cgdzpm255': 'Mixer(0.25, 5)',
    #
    'cgd101': 'MixerDif(0.10, 1)',
    'cgd102': 'MixerDif(0.10, 2)',
    'cgd103': 'MixerDif(0.10, 3)',
    'cgd104': 'MixerDif(0.10, 4)',
    'cgd105': 'MixerDif(0.10, 5)',
    'cgd106': 'MixerDif(0.10, 6)',
    'cgd107': 'MixerDif(0.10, 7)',
    'cgd152': 'MixerDif(0.15, 2)',
    'cgd153': 'MixerDif(0.15, 3)',
    'cgd154': 'MixerDif(0.15, 4)',
    'cgd155': 'MixerDif(0.15, 5)',
    'cgd156': 'MixerDif(0.15, 6)',
    'cgd157': 'MixerDif(0.15, 7)',
    'cgd201': 'MixerDif(0.20, 2)',
    'cgd202': 'MixerDif(0.20, 2)',
    'cgd203': 'MixerDif(0.20, 3)',
    'cgd204': 'MixerDif(0.20, 4)',
    'cgd205': 'MixerDif(0.20, 5)',
    'cgd206': 'MixerDif(0.20, 6)',
    'cgd207': 'MixerDif(0.20, 7)',
    'cgd253': 'MixerDif(0.25, 3)',
    'cgd303': 'MixerDif(0.30, 3)',
    #
    'cgs101': 'MixerSum(0.10, 1)',
    'cgs102': 'MixerSum(0.10, 2)',
    'cgs103': 'MixerSum(0.10, 3)',
    'cgs104': 'MixerSum(0.10, 4)',
    'cgs105': 'MixerSum(0.10, 5)',
    'cgs106': 'MixerSum(0.10, 6)',
    'cgs107': 'MixerSum(0.10, 7)',
    'cgs152': 'MixerSum(0.15, 2)',
    'cgs153': 'MixerSum(0.15, 3)',
    'cgs154': 'MixerSum(0.15, 4)',
    'cgs155': 'MixerSum(0.15, 5)',
    'cgs156': 'MixerSum(0.15, 6)',
    'cgs157': 'MixerSum(0.15, 7)',
    'cgs201': 'MixerSum(0.20, 2)',
    'cgs202': 'MixerSum(0.20, 2)',
    'cgs203': 'MixerSum(0.20, 3)',
    'cgs204': 'MixerSum(0.20, 4)',
    'cgs205': 'MixerSum(0.20, 5)',
    'cgs206': 'MixerSum(0.20, 6)',
    'cgs207': 'MixerSum(0.20, 7)',
    #
    'cgdzpd102': 'MixerDif(0.10, 2)',
    'cgdzpd103': 'MixerDif(0.10, 3)',
    'cgdzpd104': 'MixerDif(0.10, 4)',
    'cgdzpd105': 'MixerDif(0.10, 5)',
    'cgdzpd106': 'MixerDif(0.10, 6)',
    'cgdzpd107': 'MixerDif(0.10, 7)',
    'cgdzpd108': 'MixerDif(0.10, 8)',
    'cgdzpd152': 'MixerDif(0.15, 2)',
    'cgdzpd153': 'MixerDif(0.15, 3)',
    'cgdzpd154': 'MixerDif(0.15, 4)',
    'cgdzpd155': 'MixerDif(0.15, 5)',
    'cgdzpd156': 'MixerDif(0.15, 6)',
    'cgdzpd157': 'MixerDif(0.15, 7)',
    'cgdzpd158': 'MixerDif(0.15, 8)',
    'cgdzpd202': 'MixerDif(0.20, 2)',
    'cgdzpd203': 'MixerDif(0.20, 3)',
    'cgdzpd204': 'MixerDif(0.20, 4)',
    'cgdzpd205': 'MixerDif(0.20, 5)',
    'cgdzpd206': 'MixerDif(0.20, 6)',
    'cgdzpd207': 'MixerDif(0.20, 7)',
    'cgdzpd208': 'MixerDif(0.20, 8)',
    'cgdzpd252': 'MixerDif(0.25, 2)',
    'cgdzpd253': 'MixerDif(0.25, 3)',
    'cgdzpd254': 'MixerDif(0.25, 4)',
    'cgdzpd255': 'MixerDif(0.25, 5)',
    'cgdzpd256': 'MixerDif(0.25, 6)',
    'cgdzpd257': 'MixerDif(0.25, 7)',
    'cgdzpd258': 'MixerDif(0.25, 8)',
    'cgdzpd302': 'MixerDif(0.30, 2)',
    'cgdzpd303': 'MixerDif(0.30, 3)',
    'cgdzpd304': 'MixerDif(0.30, 4)',
    'cgdzpd305': 'MixerDif(0.30, 5)',
    'cgdzpd306': 'MixerDif(0.30, 6)',
    'cgdzpd307': 'MixerDif(0.30, 7)',
    'cgdzpd308': 'MixerDif(0.30, 8)',
    'cgdzpd352': 'MixerDif(0.35, 2)',
    'cgdzpd353': 'MixerDif(0.35, 3)',
    'cgdzpd354': 'MixerDif(0.35, 4)',
    'cgdzpd355': 'MixerDif(0.35, 5)',
    'cgdzpd356': 'MixerDif(0.35, 6)',
    'cgdzpd357': 'MixerDif(0.35, 7)',
    'cgdzpd358': 'MixerDif(0.35, 8)',
    'cgdzpd402': 'MixerDif(0.40, 2)',
    'cgdzpd403': 'MixerDif(0.40, 3)',
    'cgdzpd404': 'MixerDif(0.40, 4)',
    'cgdzpd405': 'MixerDif(0.40, 5)',
    'cgdzpd406': 'MixerDif(0.40, 6)',
    'cgdzpd407': 'MixerDif(0.40, 7)',
    'cgdzpd408': 'MixerDif(0.40, 8)',
    #
    'jacapo': 'dacapo',
    }

if __name__ == '__main__':

    assert len(sys.argv) > 1
    if len(sys.argv) == 2:
        taskname = sys.argv[1]
        tag = None
        runs = None
    if len(sys.argv) == 3:
        taskname = sys.argv[1]
        tag = sys.argv[2]
        runs = None
    if len(sys.argv) == 4:
        taskname = sys.argv[1]
        tag = sys.argv[2]
        runs = sys.argv[3]

    if runs is None:  # use all json files as runs
        runs = []
        for f in glob.glob(taskname + '-' + tag + '*.json'):
            runs.append(os.path.splitext(f)[0].split('_')[-1])
    else:
        runs = runs.split(',')

    labels = []
    for n, r in enumerate(runs):
        l = str(n) + ': ' + rundefs[r]
        # special cases
        if r == 'm':
            l += '\ndefault'
        elif r.startswith('inititer'):
            inititer = r[len('inititer'):len('inititer') + 2]
            if inititer == '00':
                inititer = 'None'
            else:
                inititer = str(int(inititer))
            l += '\n initial cg iter:' + inititer
        elif r.startswith('cgbands'):
            nbands = r[len('cgbands'):len('cgbands') + 2]
            if nbands == '00':
                nbands = 'None'
            else:
                nbands = str(-int(nbands))
            l += '\nnbands=' + nbands
        elif r.startswith('dzpbands'):
            nbands = r[len('dzpbands'):len('dzpbands') + 2]
            if nbands == '00':
                nbands = 'None'
            else:
                nbands = str(-int(nbands))
            l += '\nnbands=' + nbands
        elif r.startswith('bands'):
            nbands = r[len('bands'):len('bands') + 2]
            if nbands == '00':
                nbands = 'None'
            else:
                nbands = str(-int(nbands))
            l += '\nnbands=' + nbands
        elif r.startswith('szdzp'):
            l += '\nsz(dzp)'
        elif r.startswith('szpdzp'):
            l += '\nszp(dzp)'
        elif r.startswith('dzp'):
            l += '\ndzp'
        elif r.startswith('cgdzp'):
            l += '\ncg dzp'
        elif r.startswith('cg'):
            l += '\ncg'
        if 'mp' in r:
            l += '\nMethfesselPaxton'
        labels.append(l)

    steps = 80
    t = Task(taskname, ','.join(runs), labels=labels, tag=tag, steps=steps,
             tunit='h')
    t.analyse()
