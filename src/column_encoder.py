# column_encoder.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import prepare_model_for_peft, get_peft_model, LoraConfig, TaskType
from typing import List
import math
import numpy as np

class ColumnEncoder:
    """
    Handles:
      - loading a base LLM (causal)
      - optional LoRA fine-tuning (trainability via train_lora.py)
      - producing fixed-dim embeddings for column documents by pooling hidden states
    """

    def __init__(self, model_name: str, device=None, use_peft=False, peft_config=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Important for causal models to have padding token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
            load_in_4bit=False  # change if you want bitsandbytes q4
        )
        self.model.to(self.device)

        self.use_peft = use_peft
        if use_peft and peft_config:
            prepare_model_for_peft(self.model)
            self.model = get_peft_model(self.model, peft_config)

    def save_peft(self, out_dir):
        if self.use_peft:
            self.model.save_pretrained(out_dir)

    def embed_text(self, text: str, pooling: str = "mean", max_length=1024):
        """
        Tokenize `text`, run the model to get last_hidden_state, pool token embeddings to a vector.
        For causal models we use last_hidden_state from the model.forward with output_hidden_states.
        """
        enc = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # Ensure output_hidden_states
        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        # last hidden state: (batch, seq, hidden)
        # some HF causal models put hidden states in out.hidden_states[-1]
        last_hidden = out.hidden_states[-1]  # (1, seq, dim)
        attn = enc["attention_mask"].unsqueeze(-1)  # (1, seq, 1)
        last_hidden = last_hidden * attn  # zero-out padded tokens

        if pooling == "mean":
            summed = last_hidden.sum(dim=1)  # (1, dim)
            denom = attn.sum(dim=1).clamp(min=1e-9)
            vec = (summed / denom).squeeze(0).detach().cpu().numpy()
        elif pooling == "cls":
            vec = last_hidden[:, 0, :].squeeze(0).detach().cpu().numpy()
        elif pooling == "max":
            vec = (last_hidden + (attn - 1) * 1e9).max(dim=1).values.squeeze(0).detach().cpu().numpy()
        else:
            raise ValueError("Unknown pooling")
        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def batch_embed_texts(self, texts: List[str], pooling="mean", batch_size=8):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=1024, return_tensors="pt", padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                res = self.model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = res.hidden_states[-1] * enc["attention_mask"].unsqueeze(-1)
            if pooling == "mean":
                summed = last_hidden.sum(dim=1)
                denom = enc["attention_mask"].unsqueeze(-1).sum(dim=1).clamp(min=1e-9)
                vecs = (summed / denom).detach().cpu().numpy()
            else:
                raise NotImplementedError
            # normalize each
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
            vecs = vecs / norms
            out.extend(vecs.tolist())
        return out
