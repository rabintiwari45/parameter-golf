# INT2 + INT6 quantization: BPB vs model size

## Why INT2 + INT6 hurts BPB

- **INT2 bulk** only has **four** reconstruction levels per weight (vs **16** for NF4). Most parameters live in the bulk, so **most of the BPB degradation usually comes from here**, not from outliers.
- **INT6 outliers** are coarser than INT8 (fewer levels and a narrower dynamic range), so tails and large-magnitude weights are reproduced worse unless you allocate **more** of the budget to outliers (higher outlier fraction, or smaller groups so bulk error drops).

A large drop in file size with a large rise in BPB is **expected** for this mode: it is a very aggressive rate–distortion trade.

## How to lower BPB while staying INT2 + INT6

Each step below generally trades **larger file** for **better BPB**. Try in this order:

1. **Smaller group size** (usually the strongest lever for INT2)  
   The int2 default is often **`INT2_GROUP_SIZE` / `NF4_GROUP_SIZE` = 256**. Try **128**, then **64** (and only if needed **32**).  
   Smaller groups → more scales → larger checkpoint, lower BPB.

2. **More outliers**  
   Raise **`NF4_OUTLIER_PERCENT`** / **`INT2_OUTLIER_PERCENT`** (e.g. 0.15 → 0.20 → 0.25).  
   More INT6 outlier weights → better tails → larger checkpoint.

3. **Per-row outlier mask (ablate on int2)**  
   Set **`INT2_OUTLIER_PER_ROW=1`** and compare; it can help some attention/MLP layouts depending on how errors are distributed across rows.

4. **Scale storage (secondary)**  
   For int2, defaults often use fp32 scales. You can try **`INT2_SCALES_FP16=1`** with **`GROUP_SCALE_DTYPE=bf16`** to shrink scale storage; the effect on BPB is usually **small** compared to group size and outlier percentage.

5. **If BPB is the primary goal**  
   Consider keeping **NF4 bulk** and shrinking only the outlier path: **`BULK_QUANT_MODE=nf4`** with **`OUTLIER_BITS=6`**. That typically lands on a **better size vs BPB** point than full INT2 + INT6.

Also prefer **`NF4_OUTLIER_SELECT=residual`** (if enabled in your tree): same outlier **count**, often a better choice of **which** weights become outliers than magnitude-only selection.

## How to debug (map weight error → BPB)

The workflow is documented in `debug_quant_roundtrip_per_tensor` in `test_int_4.py`.

1. Load the **floating-point** `state_dict` you care about (e.g. `final_model.pt` or `base_model.state_dict()`).
2. Run:

```python
from test_int_4 import debug_quant_roundtrip_per_tensor, log_worst_quant_tensors
import torch

sd = torch.load("final_model.pt", map_location="cpu", weights_only=False)
m = debug_quant_roundtrip_per_tensor(sd)
log_worst_quant_tensors(m, top_k=30)
```

3. **Interpretation:** tensors with the largest **`rel_mse`** after quant round-trip usually correlate with BPB damage. Common suspects include **output / embedding** and specific **MLP or attention projection** layers.
4. **Ablate one knob at a time:** change only `INT2_GROUP_SIZE`, re-run; then only outlier percent; avoid changing two things at once so you know what helped.

**Optional:** for a bad layer, compare **forward logits** on a fixed batch with FP32 weights vs dequantized weights to confirm the regression is from quantization, not from eval/tokenizer setup.

## Bottom line

With **INT2 + INT6**, you should not expect NF4-level BPB without paying extra bytes. Practical recovery is **smaller groups + more outliers**, and **`debug_quant_roundtrip_per_tensor` + `log_worst_quant_tensors`** to see which layers dominate the error.
