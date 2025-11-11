# train_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_lora(model_name, docs, output_dir, epochs=1, batch_size=2, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.startswith("cuda") else None,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    # model = prepare_model_for_peft(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # prepare dataset
    from .dataset import ColumnDocDataset
    dataset = ColumnDocDataset(docs, tokenizer)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {
        "input_ids": torch.nn.utils.rnn.pad_sequence([it["input_ids"] for it in x], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([it["attention_mask"] for it in x], batch_first=True, padding_value=0)
    })

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix({"loss": float(loss.detach().cpu().numpy())})

    # save PEFT adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
