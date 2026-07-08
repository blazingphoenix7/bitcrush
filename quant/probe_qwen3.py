"""Confirm Qwen3-0.6B-Base architecture + exact tensor names/shapes before writing WGSL kernels."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-0.6B-Base"

c = AutoConfig.from_pretrained(MODEL)
print("=== CONFIG ===")
for k in ["model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
          "num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta", "rms_norm_eps",
          "hidden_act", "tie_word_embeddings", "attention_bias", "max_position_embeddings"]:
    print(f"  {k} = {getattr(c, k, 'N/A')}")

print("\n=== loading model (downloads ~1.2GB, cached after) ===", flush=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
print("params (M):", round(sum(p.numel() for p in m.parameters()) / 1e6, 1))

sd = m.state_dict()
print("\n=== tensor names & shapes (layer 0 + non-layer only) ===")
for k, v in sd.items():
    if ".layers." in k and not k.startswith("model.layers.0."):
        continue
    print(f"  {k}  {tuple(v.shape)}")
print("\ntotal tensors:", len(sd))
print("has lm_head.weight in sd:", "lm_head.weight" in sd)
