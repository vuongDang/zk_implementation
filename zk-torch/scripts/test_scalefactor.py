import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prompt = "The capital of France is"
tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]

for scale_log in [8, 10, 12]:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    scale = 2 ** scale_log
    
    with torch.no_grad():
        # original prediction
        out_original = model(**{"input_ids": tokens}).logits[0, -1]
        top_original = out_original.argmax().item()
        
        # quantize weights
        for param in model.parameters():
            param.data = torch.round(param.data * scale) / scale
        
        # quantized prediction
        out_quantized = model(**{"input_ids": tokens}).logits[0, -1]
        top_quantized = out_quantized.argmax().item()
        
        # metrics
        cos_sim = F.cosine_similarity(
            out_original.unsqueeze(0), 
            out_quantized.unsqueeze(0)
        ).item()
        
        # KL divergence between softmax distributions
        p = F.softmax(out_original, dim=0)
        q = F.softmax(out_quantized, dim=0)
        kl = F.kl_div(q.log(), p, reduction='sum').item()
        
        same_token = top_original == top_quantized
        
        print(f"scale_factor_log={scale_log}:")
        print(f"  cosine similarity: {cos_sim:.6f}")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  same top token: {same_token}")
        print(f"  original top token: '{tokenizer.decode([top_original])}'")
        print(f"  quantized top token: '{tokenizer.decode([top_quantized])}'")
        print()
