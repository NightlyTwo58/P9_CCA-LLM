from torch.utils.data import Dataset
from typing import List, Tuple

class ColumnDocDataset(Dataset):
    """
    Dataset of (column_document_text) used to fine-tune the LLM via LoRA.
    Each item is a single textual document representing a column (header + samples).
    """
    def __init__(self, docs: List[str], tokenizer, max_length=1024):
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        txt = self.docs[idx]
        enc = self.tokenizer(txt, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
