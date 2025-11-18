import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import torch
# Add BitsAndBytesConfig to your imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =================== CONFIG ===================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
chunk_size_chars = 12000
overlap_chars = 500
header_reference = "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES AND USE OF PROCEEDS"
similarity_threshold = 0.1
# ============================================

# --- Quantization Configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # Pass the quantization config
    quantization_config=bnb_config,
    # Use bfloat16 for better Llama performance (if supported)
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# ------------------ UTILITIES ------------------

def extract_cik_and_date(text):
    cik = "N/A"
    filing_date = "N/A"
    for line in text.splitlines():
        if "CIK" in line:
            cik = ''.join(filter(str.isdigit, line))
        if "Date" in line or "Filed" in line:
            filing_date = line.split()[-1]
        if cik != "N/A" and filing_date != "N/A":
            break
    return cik, filing_date

def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n")

def fuzzy_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_best_chunk(text, ref_header, chunk_size=chunk_size_chars, threshold=similarity_threshold):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    best_chunk = None
    best_score = 0

    for i, line in enumerate(lines):
        score = fuzzy_similarity(line, ref_header)
        if score > best_score:
            best_score = score
            if score >= threshold:
                start_idx = max(0, i - 5)  # include lines just above header
                best_chunk_lines = lines[start_idx:start_idx + chunk_size]
                best_chunk = "\n".join(best_chunk_lines)

    return best_chunk if best_score >= threshold else None

def generate_prompt(chunk_text):
    return (
        f"Extract the share repurchase table from the following text chunk:\n"
        f"{chunk_text}\n\n"
        "Output only a structured table with columns:\n"
        "Start Date | End Date | Class of Shares | Number of Shares | Average Price per Share | "
        "Dollar Value of Shares Repurchased | Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs | "
        "Maximum Number (or Approximate Dollar Value) of Shares that May Yet Be Purchased Under the Plans or Programs\n"
        "Fill missing data with N/A. Always include <END_OF_TABLE> at the end.\n"
        "OUTPUTT:"
    )

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.1,
                top_p=1.0,
                do_sample=True,
                repetition_penalty=1.25,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error during model.generate(): {e}")
            return ""
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "OUTPUTT:" in generated_text:
        return generated_text.split("OUTPUTT:", 1)[1].strip()
    return generated_text

def parse_extracted_table(table_text, cik, filing_date):
    lines = [l for l in table_text.split("\n") if "|" in l and "---" not in l]
    if not lines:
        return pd.DataFrame(columns=[
            "Start Date", "End Date", "Class of Shares", "Number of Shares",
            "Average Price per Share", "Dollar Value of Shares Repurchased",
            "Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs",
            "Maximum Number (or Approximate Dollar Value) of Shares that May Yet Be Purchased Under the Plans or Programs",
            "CIK", "Filing Date"
        ])
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    data = []
    for row in lines[1:]:
        values = [c.strip() for c in row.split("|")[1:-1]]
        while len(values) < len(headers):
            values.append("N/A")
        data.append(values)
    df = pd.DataFrame(data, columns=headers)
    df["CIK"] = cik
    df["Filing Date"] = filing_date
    return df

# ------------------ MAIN PROCESS ------------------

def process_all_files(input_folder, output_folder, combined_csv_path=None):
    os.makedirs(output_folder, exist_ok=True)
    all_dfs = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", "_parsed.csv"))

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                html_text = f.read()
            plain_text = html_to_text(html_text)
            cik, filing_date = extract_cik_and_date(plain_text)

            best_chunk = extract_best_chunk(plain_text, header_reference)
            if not best_chunk:
                print(f"No valid repurchase table found in {filename}, skipping.")
                continue

            prompt = generate_prompt(best_chunk)
            table_text = generate_text(prompt)
            if not table_text.strip():
                print(f"LLM returned empty output for {filename}, skipping.")
                continue

            df = parse_extracted_table(table_text, cik, filing_date)
            if df.empty:
                print(f"Failed to parse table for {filename}, skipping.")
                continue

            df.to_csv(output_path, index=False)
            all_dfs.append(df)
            print(f"Processed {filename} → {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if combined_csv_path and all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV saved at {combined_csv_path}")

# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    input_folder = "C:\\Users\\xuena\\OneDrive\\Documents\\GitHub\\P9_CCA-LLM\\data\\2011\\QTR1"
    output_folder = "C:\\Users\\xuena\\OneDrive\\Documents\\GitHub\\P9_CCA-LLM\\data\\2011_output"
    # if len(sys.argv) == 3:
    #     input_folder = sys.argv[1]
    #     output_folder = sys.argv[2]
    # else:
    #     input_folder = input("Enter input folder path: ").strip()
    #     output_folder = input("Enter output folder path: ").strip()

    combined_csv_path = os.path.join(output_folder, "all_repurchase_tables.csv")
    process_all_files(input_folder, output_folder, combined_csv_path)
