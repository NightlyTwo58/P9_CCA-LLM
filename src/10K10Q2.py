import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from main import build_column_documents, main as column_main
from utils import read_csv
import pandas as pd
import tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "C:/Users/xuena/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"

def parse_extracted_table(table_text, model_name="tiiuae/mistral-7b"):
    # Convert the markdown-style table to DataFrame
    lines = [l for l in table_text.split("\n") if "|" in l and "---" not in l]
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    data = []
    for row in lines[1:]:
        values = [c.strip() for c in row.split("|")[1:-1]]
        data.append(values)
    df = pd.DataFrame(data, columns=headers)

    # Save temporary CSV
    tmp_dir = tempfile.mkdtemp()
    tmp_csv = os.path.join(tmp_dir, "table.csv")
    df.to_csv(tmp_csv, index=False)

    # Build embeddings with your parser
    class Args:
        csv = "table.csv"
        model_name = model_path
        sample_per_col = 100
        max_examples = 50
        embed_batch_size = 2
        cells_per_col = 20
        k = 3
        out_dir = os.path.join(tmp_dir, "out_embs")
        project_dim = 256
        cpu = True
        seed = 42

    args = Args()
    column_main(args)
    return df

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)


def generate_prompt(file_text):
    return (
        f"Extract the share repurchase details from the table in the following financial filing text:\n"
        f"\"{file_text}\".\n\n\n"
        "Strictly output only the structured table in the exact format below, with no additional text, notes, or commentary:\n\n"
        "CIK Number: <Extracted_CIK>\n\n"
        "Start Date | End Date | Class of Shares | Number of Shares | Average Price per Share | Dollar Value of Shares Repurchased | Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs | Maximum Number (or Approximate Dollar Value) of Shares that May Yet Be Purchased Under the Plans or Programs\n"
        "-----------|-----------|----------------|------------------|------------------------|-----------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------\n"
        "<YYYY-MM-DD> | <YYYY-MM-DD> | <Class> | <Number> | <Price> | <Dollar Value> | <Number from Plan> | <Remaining Capacity>\n"
        "<YYYY-MM-DD> | <YYYY-MM-DD> | <Class> | <Number> | <Price> | <Dollar Value> | <Number from Plan> | <Remaining Capacity>\n"
        "...\n"
        "<END_OF_TABLE>\n\n"
        "### **Instructions:**\n"
        "- If a date is a range (e.g., 'July 2023' or 'Q2 2023'), extract a start and end date. Keep format as YYYY-MM-DD\n"
        "- If only one date is available, repeat it in both 'Start Date' and 'End Date'.\n"
        "- If data is missing or not found, write 'N/A'.\n"
        "- Extract the CIK Number from the document and place it at top.\n"
        "- Always include the <END_OF_TABLE> marker at the end of the table.\n"
        "- Maintain the exact column names and format with separators | as shown above to ensure CSV compatibility.\n"
        "- Output only the table content: do not include notes, instructions, or any other text."
        "OUTPUTT:"
    )

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.1,
            top_p=1.0,
            do_sample=True, repetition_penalty=1.25, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the content after 'OUTPUTT:'
    if "OUTPUTT:" in generated_text:
        generated_text = generated_text.split("OUTPUTT:", 1)[1].strip()

    return generated_text

def process_all_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".txt", "_parsed.csv"))

            if os.path.exists(output_path):
                print(f"Skipping (already exists): {filename}")
                continue

            try:
                with open(input_path, "r", encoding="utf-8") as file:
                    file_text = file.read()

                # Step 1: LLM extraction
                prompt = generate_prompt(file_text)
                table_text = generate_text(prompt)

                # Step 2: Context-aware parsing
                df = parse_extracted_table(table_text)

                df.to_csv(output_path, index=False)
                print(f"Processed and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Main: expects two arguments from SLURM script - input and output folder paths
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python testing2_code.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    os.makedirs(output_folder, exist_ok=True)
    process_all_files(input_folder, output_folder)