## Tutorial of modifying protein generator
### Step 1: locate the sample function
The sample function is located in **./utils/diff_utils.py**, this file is imported to **./utils/sampler.py**.
```python
def take_step_nostate(...):
    ...
```
### Step 2: import necessary functions to the file
At the beginning of the file, import related protein watermark functions. ***Remeber to replace the private key with your own key!*** 
```python
######################################
### IMPORT WATERMARK PACKAGE #########
######################################
from proteinwatermark import (
    DeltaGumbel_Reweight,
    WatermarkLogitsProcessor,
    WatermarkDetector
)

delta_wp = WatermarkLogitsProcessor(
    b"private key", #remember to replace it with your own private key!
    DeltaGumbel_Reweight(),
    context_code_length=5,
)
######################################
```

### Step 3: modify the sampling process to add watermark
In protein generator, probability of the whole sequence is generated at each diffusion step. The original model uses argmax to sample sequence from the probability. Here, we modify this process to sample watermarked sequence at each step.
```python
def take_step_nostate(...):
    ...
    with torch.no_grad():
        ...
        logit_aa_s = logit_aa_s.reshape(B,-1,L)
        ### ADD WATERMARK ##########################################################
        ############################################################################
        seq_out = torch.zeros(B, L, dtype=torch.int64, device=logit_aa_s.device) #make sure the dtype is int64/long, otherwise, there may exist bugs.
        # though the full probability is given,
        # treat it as one-by-one generation process
        for i in range(logit_aa_s.shape[-1]):
            current_logit = logit_aa_s[:,:,i].reshape(B, -1)
            seq_record = seq_out.detach().cpu().numpy()
            
            current_logit = delta_wp("order_agnoistic", 
                                     seq_record, # the context information
                                     current_logit.detach().cpu().numpy(),
                                     current_pos=i # the context information)
            current_logit = torch.FloatTensor(current_logit).to(logit_aa_s.device)
            
            probs = torch.softmax(current_logit, dim=-1)
            token_t = torch.multinomial(probs, 1)
            seq_out[:, i] = token_t
        ############################################################################
        ...
```
