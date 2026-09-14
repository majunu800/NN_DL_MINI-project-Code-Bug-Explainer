# Mini Project Report

## Project Title

**PyExplain: Python Code Bug Explainer Using Static Analysis and Large Language Models**

## 1. Problem Statement

Python programming errors can be difficult for beginners to understand. A syntax error may stop a program from running, while a logical error may produce an incorrect result without showing an obvious error message. Existing error messages often identify the location of a problem but do not clearly explain the cause or suggest how to fix it.

This project develops a tool that accepts Python code, detects common syntax and logical problems, and provides understandable explanations and repair suggestions. The project also includes an optional fine-tuning pipeline for adapting a smaller open-source Large Language Model to Python debugging tasks.

## 2. Objective

The main objectives of this project are:

- To detect syntax errors in Python code.
- To identify likely logical errors such as undefined variables.
- To provide simple explanations and possible fixes.
- To create an easy-to-use web interface for students and beginner programmers.
- To prepare a debugging dataset from Hugging Face.
- To provide a QLoRA fine-tuning pipeline for a smaller open-source coding model.
- To combine deterministic Python analysis with optional AI-generated explanations.

## 3. Existing System

In the existing approach, programmers generally depend on:

- Python interpreter error messages.
- Manual debugging and print statements.
- Integrated Development Environment diagnostics.
- Online forums and documentation.
- General-purpose AI assistants.

These approaches have some limitations. Error messages may be difficult for beginners to understand, manual debugging takes time, and general-purpose models may provide inconsistent explanations. Many tools also focus mainly on syntax and do not provide a complete explanation of the likely cause and correction.

## 4. Novelty of the Proposed System

The proposed system combines traditional program analysis with an optional fine-tuned language model.

The novel features are:

- Python AST-based syntax validation.
- Detection of possible undefined variables.
- Basic detection of risky `eval()` usage.
- Human-readable explanations and repair suggestions.
- A Gradio web interface for interactive debugging.
- A complete Hugging Face dataset preparation pipeline.
- A QLoRA training script for a smaller open-source coding model.
- Static analysis results remain available even when the language model is not installed or unavailable.

This hybrid approach is useful because static analysis provides fast and predictable checks, while the language model can provide more detailed natural-language explanations after fine-tuning.

## 5. Code Implementation

### 5.1 Technologies Used

- **Python 3.13**
- **Python `ast` module** for syntax and code analysis
- **Gradio** for the web interface
- **Hugging Face Datasets** for dataset loading
- **Transformers** for language model loading
- **PEFT and TRL** for LoRA and supervised fine-tuning
- **BitsAndBytes** for 4-bit quantization during QLoRA training
- **Qwen2.5-Coder-1.5B-Instruct** as the recommended base model

### 5.2 Project Files

- `app.py`: Gradio interface, static analyzer, and optional model inference.
- `prepare_dataset.py`: Downloads and normalizes debugging data into JSONL format.
- `train.py`: Fine-tunes the base coding model using QLoRA.
- `data/sample_debugging.jsonl`: Small sample dataset for testing.
- `test_app.py`: Basic analyzer tests.
- `requirements.txt`: Python dependencies.

### 5.3 Processing Flow

```text
Python code input
       |
       v
Python AST parser
       |
       +--> Syntax error detection
       +--> Undefined-name detection
       +--> Risk detection
       |
       v
Readable diagnosis and repair suggestion
       |
       +--> Optional fine-tuned LLM explanation
       |
       v
Gradio output screen
```

### 5.4 Running the Project

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the application:

```powershell
python app.py
```

Open the local web address:

```text
http://127.0.0.1:7861
```

### 5.5 Optional Fine-Tuning Pipeline

Prepare a debugging dataset:

```powershell
python prepare_dataset.py --limit 5000
```

Train the QLoRA adapter on a CUDA-enabled machine:

```powershell
python train.py --data data/debugging.jsonl --epochs 1
```

Run the application with the trained model:

```powershell
$env:DEBUGGER_MODEL_PATH = "outputs/code-debugger"
python app.py
```

The static analyzer was tested locally. Full QLoRA training was not executed on the development computer because it has CPU-only PyTorch and no CUDA GPU.

## 6. Output

### Example 1: Syntax Error

Input:

```python
for number in numbers
    print(number)
```

Output:

```text
Diagnosis

Syntax error (line 1)
What happened: Python could not understand this line because it expected ':'.
Why it matters: A syntax error stops the program before it can run.
Recommended fix: Add a colon (:) at the end of the for-loop statement.
```

Corrected code:

```python
for number in numbers:
    print(number)
```

### Example 2: Possible Logical Error

Input:

```python
print(total)
```

Output:

```text
Diagnosis

Possible logic error (line 1)
What happened: The name total is used here, but Python cannot find where it was created.
Why it matters: The program may stop with a NameError, or the variable name may be misspelled.
Recommended fix: Check the spelling and create total before line 1.
```

### Example 3: Valid Code

Input:

```python
def greet(name):
    return name
```

Output:

```text
Diagnosis

Looks good
What happened: Python can read this code, and the basic checks did not find an obvious problem.
Why it matters: Run the program with real examples too. Static checks cannot prove that every result is correct.
```

### Interface Output

The Gradio interface uses a colorful two-panel layout. The left panel accepts Python code and has Debug code and Clear buttons. The right panel displays the diagnosis in a consistent format: **What happened**, **Why it matters**, and **Recommended fix**. It can be accessed locally at `http://127.0.0.1:7861` while the application is running.

## 7. GitHub Repository Link

**GitHub repository:** https://github.com/majunu800/NN_DL_MINI-project-Code-Bug-Explainer

## Conclusion

PyExplain provides a practical beginner-friendly tool for understanding Python programming errors. The current implementation combines fast static analysis with a scalable QLoRA fine-tuning route. The static analyzer and Gradio interface work locally without requiring a GPU, while the optional model-training pipeline can be executed on a CUDA-enabled system for richer debugging explanations.
