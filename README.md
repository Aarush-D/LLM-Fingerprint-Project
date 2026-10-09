# LLM Fingerprint Project

CMPSC 448 Midterm Project: **Who Wrote It? Identifying LLMs from Their Responses**

## Goal

Train a CNN and an LSTM to predict which LLM generated a text response.

Suggested classes:
- GPT
- Claude
- Gemini

## Folder Structure

- `data/prompts.csv` - 90 prompts across six categories
- `data/llm_dataset_template.csv` - blank template for collecting model responses
- response CSVs in `data/` - collected/sample response data; keep these, and save the final complete 3-family dataset as `data/llm_dataset.csv`
- `src/preprocess.py` - tokenization and vocabulary helpers
- `src/models.py` - CNN and LSTM models
- `src/train.py` - training/evaluation program
- `results/` - model outputs, metrics, and graphs
- `report/report_outline.md` - report structure

## Step 1: Create a Python environment

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Collect Responses

Open `data/prompts.csv`.

For every prompt, collect one response from each of:
- GPT
- Claude
- Gemini

Put those responses into `data/llm_dataset_template.csv`.

Each prompt should appear three times, once per LLM.

Save the completed file as:

```text
data/llm_dataset.csv
```

Required columns:

```text
prompt_id,category,LLM_name,LLM_Input,LLM_output
```

## Step 3: Run the Required Experiments

Go into the source folder:

```bash
cd src
```

### CNN

```bash
python train.py --model cnn --mode output
python train.py --model cnn --mode input
python train.py --model cnn --mode both
```

### LSTM

```bash
python train.py --model lstm --mode output
python train.py --model lstm --mode input
python train.py --model lstm --mode both
```

This gives the six experiments needed for RQ1 and RQ2.

## Step 4: Review Results

Look inside:

```text
results/
```

Each run will produce:
- model file
- training history CSV
- loss graph
- text summary with test accuracy, classification report, and confusion matrix

## Important

Do not invent results. Only put numbers in the report after running the experiments.
# LLM-Fingerprint-Project
