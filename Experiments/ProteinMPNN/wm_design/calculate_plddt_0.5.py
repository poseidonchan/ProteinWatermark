#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import esm
import torch
from tqdm import tqdm
import biotite.structure.io as bsio


# In[2]:


directory = "./wm_outputs/monomer_wm_0.5/seqs/"


# In[3]:


model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()


# In[4]:


def average_pLDDT(seq):
    with torch.no_grad():
        output = model.infer_pdb(seq)
    with open(directory+"result.pdb", "w") as f:
        f.write(output)
    struct = bsio.load_structure(directory+"result.pdb", extra_fields=["b_factor"])
    return struct.b_factor.mean()


# In[ ]:


for filename in os.listdir(directory):
    if filename.endswith('.fa'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # Process the designed sequences
        output_lines = []
        print(filename)
        for i in tqdm(range(len(lines))):
            line = lines[i].strip()
            if line.startswith('>T='):
                if "pLDDT" in line:
                    output_lines.append(line+ '\n')
                else:
                    # Extract the designed sequence
                    sequence = lines[i + 1].strip()
                    # Evaluate the designed sequence
                    if len(sequence) <= 750:
                        plddt_score = average_pLDDT(sequence)
                        header = line + f', pLDDT={plddt_score}'
                    else:
                        header = line + ', pLDDT=NaN'
                    # Add the pLDDT score to the header
                    # header = line + f', pLDDT={plddt_score}'
                    output_lines.append(header + '\n')
                    output_lines.append(sequence + '\n')
            elif line.startswith('>'):
                output_lines.append(line+ '\n')
            elif i == 1:
                output_lines.append(line+ '\n')
        
        # Write the modified sequences to a new file
        with open(file_path, 'w') as file:
            file.writelines(output_lines)



