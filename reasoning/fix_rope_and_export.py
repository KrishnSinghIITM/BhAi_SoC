"""
fix_rope_and_export.py
1-Click Fix for Llama 3.2 RoPE Attention Scrambling:
Corrects Hugging Face -> Meta Rotary Position Embedding (RoPE) tensor permutations
for Q (wq) and K (wk) projection matrices across all 16 transformer layers.
"""

import os
import json
import torch
from safetensors.torch import load_file

def permute_hf_to_meta(w, n_heads, dim1, dim2):
    """Reverses Hugging Face's complex RoPE interleaving back to Meta's native layout."""
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def main():
    model_dir = "/mnt/storage/krishn/ptd/models/Llama-3.2-1B-Instruct"
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    
    if not os.path.exists(safetensors_path):
        print(f"[!] Error: {safetensors_path} not found!")
        return

    print("=" * 70)
    print(" FIXING LLAMA 3.2 1B WEIGHTS: UNPERMUTING ROPE (wq & wk)")
    print("=" * 70)
    print("[1/3] Loading raw model.safetensors...")
    weights = load_file(safetensors_path, device="cpu")

    meta_sd = {}
    for name, param in weights.items():
        new_name = name.removeprefix("model.")
        if new_name == "embed_tokens.weight":
            meta_sd["tok_embeddings.weight"] = param
            meta_sd["output.weight"] = param.clone()  # Tied embedding head
        elif new_name == "lm_head.weight":
            meta_sd["output.weight"] = param
        elif new_name.startswith("layers."):
            parts = new_name.split(".")
            layer_idx = parts[1]
            module = parts[2]
            param_name = parts[3]

            if module == "self_attn":
                if param_name == "q_proj":
                    # Llama 3.2 1B: dim1=2048, dim2=2048, n_heads=32
                    meta_sd[f"layers.{layer_idx}.attention.wq.weight"] = permute_hf_to_meta(param, 32, 2048, 2048)
                elif param_name == "k_proj":
                    # Llama 3.2 1B: dim1=512, dim2=2048, n_heads=8
                    meta_sd[f"layers.{layer_idx}.attention.wk.weight"] = permute_hf_to_meta(param, 8, 512, 2048)
                elif param_name == "v_proj":
                    meta_sd[f"layers.{layer_idx}.attention.wv.weight"] = param
                elif param_name == "o_proj":
                    meta_sd[f"layers.{layer_idx}.attention.wo.weight"] = param
            elif module == "mlp":
                if param_name == "gate_proj":
                    meta_sd[f"layers.{layer_idx}.feed_forward.w1.weight"] = param
                elif param_name == "down_proj":
                    meta_sd[f"layers.{layer_idx}.feed_forward.w2.weight"] = param
                elif param_name == "up_proj":
                    meta_sd[f"layers.{layer_idx}.feed_forward.w3.weight"] = param
            elif module == "input_layernorm":
                meta_sd[f"layers.{layer_idx}.attention_norm.weight"] = param
            elif module == "post_attention_layernorm":
                meta_sd[f"layers.{layer_idx}.ffn_norm.weight"] = param
        elif new_name == "norm.weight":
            meta_sd["norm.weight"] = param

    out_pth = os.path.join(model_dir, "consolidated.00.pth")
    print(f"[2/3] Writing corrected weights to {out_pth}...")
    torch.save(meta_sd, out_pth)

    print("[3/3] SUCCESS! Attention weights unpermuted. Model is ready for ExecuTorch compile.")
    print("=" * 70)

if __name__ == "__main__":
    main()
