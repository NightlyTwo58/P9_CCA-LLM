import pandas as pd
from typing import List
from pathlib import Path
from tqdm import tqdm

# Example using HuggingFace transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def extract_tables_from_item(text: str, model_name="tiiuae/mistral-7b") -> List[pd.DataFrame]:
    """
    Extracts all tables from Part II Item text using an LLM.

    Args:
        text: The raw text of Part II Item 2 (10-Q) or Item 5 (10-K)
        model_name: HuggingFace LLM model

    Returns:
        List of pandas DataFrames representing each table
    """
    # Initialize LLM pipeline (text-generation)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Prompt for table extraction
    prompt = f"""
You are an expert at extracting financial tables from SEC filings.
Extract all tables from the following text. Return each table in CSV format.
- Preserve all columns as headers.
- If a cell contains a list, join values with semicolons.
- Separate tables clearly using '---TABLE---'.

Text:
{text[:4000]}  # LLM context limit; may need chunking for long texts
"""

    # Generate LLM output
    output = generator(prompt, max_new_tokens=4000, do_sample=False)[0]["generated_text"]

    # Split tables
    raw_tables = output.split("---TABLE---")
    tables = []

    for table_text in raw_tables:
        # Convert CSV-like text to DataFrame
        try:
            df = pd.read_csv(pd.compat.StringIO(table_text.strip()))
            tables.append(df)
        except Exception as e:
            print(f"Warning: Could not parse table:\n{table_text[:200]}...\nError: {e}")

    return tables

def process_filing(filing_path: str, cik: str, filing_date: str, item_number: str, out_dir: str):
    """
    Process a single 10-Q or 10-K filing: extract tables and save CSVs.

    Args:
        filing_path: Path to raw text of filing
        cik: Company CIK
        filing_date: Filing date in YYYY-MM-DD
        item_number: '2' for 10-Q, '5' for 10-K
        out_dir: Directory to save CSVs
    """
    with open(filing_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Naive Part II Item extraction (can be improved with LLM)
    import re
    pattern = rf"(?i)Part\s+II.*Item\s+{item_number}.*?(?=Item\s+\d|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        print(f"Warning: Could not locate Part II Item {item_number} in {filing_path}")
        return

    item_text = match.group(0)

    # Extract tables
    tables = extract_tables_from_item(item_text)

    # Save CSVs
    os.makedirs(out_dir, exist_ok=True)
    for idx, df in enumerate(tables):
        out_path = Path(out_dir) / f"{cik}_{filing_date}_item{item_number}_table{idx+1}.csv"
        df["CIK"] = cik
        df["filing_date"] = filing_date
        df.to_csv(out_path, index=False)

# Example usage:
# process_filing("sample_10q.txt", cik="0000320193", filing_date="2025-02-01", item_number="2", out_dir="./out_tables")
