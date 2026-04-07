import io
import zlib
import torch

# -----------------------------
# DEQUANTIZATION (same as before)
# -----------------------------

def dequantize_state_dict_int8(obj: dict):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]

        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype)
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype)

    for name, t in obj["passthrough"].items():
        out_t = t.clone()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype))
        out[name] = out_t

    return out


# -----------------------------
# MAIN LOADER
# -----------------------------

def load_model(model_class, model_kwargs, path, mode="auto"):
    """
    mode:
        - "auto" → detect from extension
        - "pt" → load raw model
        - "int8" → load quantized
    """

    if mode == "auto":
        if path.endswith(".ptz"):
            mode = "int8"
        elif path.endswith(".pt"):
            mode = "pt"
        else:
            raise ValueError("Unknown file type")

    print(f"Loading model: {path} (mode={mode})")

    # -----------------------------
    # LOAD RAW PT MODEL
    # -----------------------------
    if mode == "pt":
        state_dict = torch.load(path, map_location="cpu")

    # -----------------------------
    # LOAD INT8 MODEL
    # -----------------------------
    elif mode == "int8":
        with open(path, "rb") as f:
            compressed = f.read()

        decompressed = zlib.decompress(compressed)
        quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")

        state_dict = dequantize_state_dict_int8(quant_state)

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # -----------------------------
    # BUILD MODEL
    # -----------------------------
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("✅ Model loaded on CPU")
    return model, state_dict


# -----------------------------
# OUTLIER INSPECTION
# -----------------------------

def inspect_outliers(state_dict, topk=5):
    print("\n🔍 Inspecting weight outliers...\n")

    for name, w in state_dict.items():
        if not torch.is_floating_point(w):
            continue

        w_flat = w.view(-1).float()

        max_val = w_flat.abs().max().item()
        mean = w_flat.mean().item()
        std = w_flat.std().item()

        # top-k extreme values
        top_vals = torch.topk(w_flat.abs(), k=min(topk, w_flat.numel())).values
        breakpoint()

        print(f"{name}")
        print(f"  shape: {tuple(w.shape)}")
        print(f"  max|w|: {max_val:.4f}")
        print(f"  mean: {mean:.6f}, std: {std:.6f}")
        print(f"  top-{topk} abs vals: {top_vals.tolist()}")
        print("-" * 50)


# -----------------------------
# USAGE
# -----------------------------

if __name__ == "__main__":
    from train_gpt import GPT, Hyperparameters

    args = Hyperparameters()

    model_kwargs = dict(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )

    # 🔁 CHANGE THIS PATH
    path = "final_model.pt"         # OR "final_model.int8.ptz"

    model, state_dict = load_model(GPT, model_kwargs, path)

    # 🔍 Inspect outliers
    inspect_outliers(state_dict)