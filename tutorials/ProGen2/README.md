## Tutorial for modifying the ProGen2
### Step 1: locate the sampling step
For language models built on the top of transformers library, the main inference part involves the function:
```python
... = model.generate(
          input_ids,
          do_sample=True,
          temperature=temp,
          max_length=max_length,
          top_p=top_p,
          num_return_sequences=num_return_sequences,
          pad_token_id=pad_token_id,
          logits_warper=LogitsProcessorList([delta_wp]),
      )
```

Here, the long autoregressive sampling procedure is wrapped in this function, we can modify the logits by pass our own ***logins_warper*** into this function. The detailed position is in:

```python
def sample(...):
    with torch.no_grad():
        ...
        ... = model.generate(...)
    ...
...
```
### Step 2: import watermark functions
First, we need to import the watermark logit processor
```python
from proteinwatermark import transformerWatermarkLogitsProcessor, DeltaGumbel_Reweight
```
Then we need to import the patch for transformers models. The patch is available at this [repo](https://github.com/xiaoniu-578fa6bff964d005/UnbiasedWatermark/blob/cfb843e4dd0123e4997fa8a6d67673355de74fe9/unbiased_watermark/monkeypatch.py).

```python
# refer to https://github.com/xiaoniu-578fa6bff964d005/UnbiasedWatermark/tree/master
# import the patch for the easy use of transformers library
from unbiased_watermark import (
    patch_model,
)
```

### Step 3: modify the logits
First, we add a patch to current model
```python
def create_model(ckpt, fp16=True):
    if fp16:
        model = ProGenForCausalLM.from_pretrained(
            ckpt, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        model = ProGenForCausalLM.from_pretrained(ckpt)

    patch_model(model)
    return model
```
Then we modify the logits.
```python
def sample(
    ...
):
    # add the logits processor here or in the outer name space
    delta_wp = transformerWatermarkLogitsProcessor(
        b"private key", # Remeber to modify it to your own private key!
        DeltaGumbel_Reweight(),
        5,
    )
    
    with torch.no_grad():
        input_ids = ...
        tokens_batch = model.generate(
            input_ids,
            do_sample=True,
            temperature=temp,
            max_length=max_length,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            logits_warper=LogitsProcessorList([delta_wp]), # pass the processor to the generation function
        )
        ...
```

