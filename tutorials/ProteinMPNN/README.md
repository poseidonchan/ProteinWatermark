## Tutorial of modifying ProteinMPNN
### Step 1: locate the sampling step
In ProteinMPNN, the main sampling function is located in the **protein_mpnn_utils.py** file. The function name is **sample()** or **tied_sample()**.

```python
class ProteinMPNN():
  ...
  def sample(self,...):
    ...
```
### Step 2: import protein watermark functions
At the beginning of the file, import necessary functions, of note, the ***private key should be changed to your own key***.
```python
######################################
### IMPORT WATERMARK PACKAGE #########
######################################
from proteinwatermark import (
    DeltaGumbel_Reweight,
    WatermarkLogitsProcessor,
)

delta_wp = WatermarkLogitsProcessor(
    b"private key",
    DeltaGumbel_Reweight(),
    context_code_length=5,
)
######################################
```
### Step 3: add watermark to the generated logits
Then, use the logit processor to modify the logits in the sample function.
```python
def sample(self, ...):
    ...
    for t_ in range(N_nodes):
        ...
        if ...:
            ...
        else:
            ...
            logits = self.W_out(h_V_t) / temperature
            logits = logits-constant[None,:]*1e8+constant_bias[None,:]/temperature+bias_by_res_gathered/temperature
            #################################################################
            ### MODIFY THE LOGITS TO ADD WATERMARK ##########################
            S_record = S.detach().cpu().numpy() # CURRENT GENERATED SEQUENCES
            logits = delta_wp("order_agnoistic", # since it is proteinMPNN
                              S_record, # current sequences
                              logits.detach().cpu().numpy(),
                              current_pos=t.long().detach().cpu())
            logits = torch.FloatTensor(logits).to(device)
            #################################################################
            ...
    ...
```
### (Optional) step 4: include modified file into the main inference function

