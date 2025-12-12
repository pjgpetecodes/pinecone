from typing import Dict

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeHelper:
    """Generate sparse vectors using SPLADE (bag-of-words expansion)."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def text_to_sparse(self, text: str) -> Dict[int, float]:
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs).logits.squeeze(0)  # [seq_len, vocab]

        # SPLADE: max over tokens of log(1 + relu(logits))
        relu = torch.relu(out)
        sp = torch.log1p(relu).max(dim=0).values  # [vocab]

        # Build sparse dict of nonzeros
        nonzero = torch.nonzero(sp, as_tuple=False).squeeze(1)
        values = sp[nonzero].cpu().tolist()
        indices = nonzero.cpu().tolist()

        # Normalize values to [0,1]
        if values:
            max_val = max(values)
            values = [v / max_val for v in values]

        return {int(i): float(v) for i, v in zip(indices, values)}
