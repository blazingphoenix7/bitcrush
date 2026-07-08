"""Export Qwen3-0.6B (instruct/post-trained) for the browser engine. Same architecture as the base
model, so the WGSL engine + tokenizer are unchanged. Overwrites web/weights-qwen3/. Also captures the
chat-template prefix/suffix token ids (thinking OFF) so the browser can wrap the user's prompt."""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "Qwen/Qwen3-0.6B"
OUT = os.path.join("web", "weights-qwen3")
TOK = os.path.join("web", "tok-qwen3")
PROMPT = "What is the meaning of life?"
N_NEW = 24


def wrap_ids(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}],
                                add_generation_prompt=True, enable_thinking=False, tokenize=False)
    return tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOK, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    for p in model.parameters():
        p.data.copy_(p.data.half().float())

    sd = model.state_dict()
    keep = [k for k in sd if k != "lm_head.weight"]
    rope_theta = float(cfg.to_dict().get("rope_theta") or 1000000.0)
    manifest = {
        "model": MODEL,
        "config": {
            "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size, "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim, "rope_theta": rope_theta, "rms_norm_eps": float(cfg.rms_norm_eps),
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        },
        "tensors": [],
    }
    buf = bytearray()
    for k in keep:
        arr = sd[k].detach().cpu().numpy().astype(np.float16)
        b = arr.tobytes()
        manifest["tensors"].append({"name": k, "shape": list(arr.shape), "dtype": "f16", "offset": len(buf), "bytes": len(b)})
        buf += b
    with open(os.path.join(OUT, "qwen3.bin"), "wb") as f:
        f.write(buf)
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print("weights: %.0f MB, %d tensors" % (len(buf) / 1e6, len(keep)))

    data = json.loads(tok.backend_tokenizer.to_str())
    vocab = data["model"]["vocab"]
    merges = [m if isinstance(m, str) else ("%s %s" % (m[0], m[1])) for m in data["model"]["merges"]]
    json.dump(vocab, open(os.path.join(TOK, "vocab.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(TOK, "merges.txt"), "w", encoding="utf-8").write("\n".join(merges))
    json.dump(data.get("pre_tokenizer", {}), open(os.path.join(TOK, "pretokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("tokenizer: vocab %d, merges %d" % (len(vocab), len(merges)))

    # chat-template prefix/suffix via a marker word
    marker = "ZQXMARKQZ"
    mk_ids = tok(marker, add_special_tokens=False).input_ids
    wrapped = wrap_ids(tok, marker)
    prefix_ids, suffix_ids = [], []
    for i in range(len(wrapped) - len(mk_ids) + 1):
        if wrapped[i:i + len(mk_ids)] == mk_ids:
            prefix_ids = wrapped[:i]
            suffix_ids = wrapped[i + len(mk_ids):]
            break

    ids = torch.tensor([wrap_ids(tok, PROMPT)])
    last = model(ids).logits[0, -1].float()
    topv, topi = torch.topk(last, 5)
    top5 = [{"id": int(i), "logit": round(float(v), 4), "tok": tok.decode([int(i)])} for v, i in zip(topv, topi)]
    gen = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    new_ids = gen[0, ids.shape[1]:].tolist()
    hs = model(ids, output_hidden_states=True).hidden_states
    hs[0][0].float().numpy().astype(np.float32).tofile(os.path.join(OUT, "ref_embed.bin"))
    hs[1][0].float().numpy().astype(np.float32).tofile(os.path.join(OUT, "ref_layer0.bin"))
    last.numpy().astype(np.float32).tofile(os.path.join(OUT, "ref_logits.bin"))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = list({tok.eos_token_id, im_end} - {None})
    ref = {"prompt": PROMPT, "templated_text": tok.decode(ids[0]), "input_ids": ids[0].tolist(),
           "chat_prefix_ids": prefix_ids, "chat_suffix_ids": suffix_ids, "eos_ids": eos_ids,
           "greedy_new_ids": new_ids, "greedy_text": tok.decode(new_ids), "top5_next": top5,
           "logits_file": "ref_logits.bin", "logits_len": int(last.shape[0]),
           "embed_file": "ref_embed.bin", "layer0_file": "ref_layer0.bin",
           "seq_len": int(ids.shape[1]), "hidden": cfg.hidden_size}
    json.dump(ref, open(os.path.join(OUT, "reference.json"), "w", encoding="utf-8"), indent=2)
    print("templated:", repr(ref["templated_text"]))
    print("prefix_ids:", prefix_ids)
    print("suffix_ids:", suffix_ids)
    print("eos_ids:", eos_ids)
    print("greedy_text:", repr(ref["greedy_text"]))


if __name__ == "__main__":
    main()
