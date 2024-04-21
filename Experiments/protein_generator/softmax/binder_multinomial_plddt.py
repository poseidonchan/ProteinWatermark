#!/usr/bin/env python
# coding: utf-8
import os
import esm
import torch
import pandas as pd
from tqdm import tqdm
import biotite.structure.io as bsio


model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

filename = './binder_multinomial_softmax.csv'
data = pd.read_csv(filename,index_col=0)
def average_pLDDT(seq):
    with torch.no_grad():
        output = model.infer_pdb(seq)
    with open("binder_multinomial_result.pdb", "w") as f:
        f.write(output)
    struct = bsio.load_structure("binder_multinomial_result.pdb", extra_fields=["b_factor"])
    return struct.b_factor.mean()

plddts = []
for i in tqdm(range(len(data))):
    plddts.append(average_pLDDT(data.sequence[i]))

data['ESMfold_plddt'] = plddts
data.to_csv(filename)

import scipy.stats
print(scipy.stats.pearsonr(data.iloc[:244, 3].values, plddts))

for i in plddts:
    print(i)