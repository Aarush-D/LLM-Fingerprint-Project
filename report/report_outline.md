# Who Wrote It? Identifying LLMs from Their Responses

## Team
- Team leader: Aarush Durgavarjhula
- Members: Aarush Durgavarjhula

## 1. Introduction
Explain the goal: determine whether a machine-learning classifier can identify which LLM generated a response.

## 2. Research Questions
### RQ1
Can we identify which LLM generated a response using only LLM output?

### RQ2
Does the user's prompt help identify the LLM?
Compare:
- Input only
- Output only
- Input + Output

## 3. Dataset Curation
Document:
- Which LLMs were used
- How many prompts were used
- Prompt categories
- How responses were collected
- How the classes were balanced
- Any cleaning performed

## 4. Methodology
### 4.1 Preprocessing
Explain tokenization, vocabulary creation, padding, and train/validation/test split.

### 4.2 CNN
Describe embedding, 1D convolution, pooling, and final classifier.

### 4.3 LSTM
Describe embedding, recurrent processing, hidden state, and final classifier.

### 4.4 Training
Document:
- optimizer
- learning rate
- epochs
- batch size
- loss function
- hardware used

## 5. Results
Include:
- CNN and LSTM accuracy
- Input-only results
- Output-only results
- Input + Output results
- Confusion matrices
- Loss curves

## 6. Discussion
Discuss:
- Which model performed better?
- Which LLMs were easiest/hardest to distinguish?
- Did the prompt improve performance?
- Why might input-only classification work or fail?

## 7. Optional Extra-Credit Analysis
Possible RQ4 analysis:
- response length
- vocabulary
- lexical diversity
- formatting
- bullet points / headings

## 8. Lessons and Experience
Explain what you learned from:
- data collection
- text preprocessing
- CNNs
- RNN/LSTMs
- evaluation
- limitations

## 9. Conclusion
Summarize the main findings and limitations.
