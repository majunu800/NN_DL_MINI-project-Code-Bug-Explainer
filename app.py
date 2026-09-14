"""Gradio UI for syntax checks and optional fine-tuned model explanations."""

import ast
import os
import re
from dataclasses import dataclass

@dataclass
class Finding:
    kind: str
    message: str
    line: int | None = None
    hint: str = ""


def static_findings(code: str) -> list[Finding]:
    if not code.strip():
        return [Finding("Input", "Paste Python code into the editor.")]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        line = error.lineno
        detail = error.msg
        hint = "Check the indicated line for missing punctuation, indentation, or unmatched brackets."
        return [Finding("Syntax error", detail, line, hint)]

    findings: list[Finding] = []
    assigned: set[str] = set()
    loaded: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
            assigned.update(argument.arg for argument in node.args.args)
            assigned.update(argument.arg for argument in node.args.kwonlyargs)
            if node.args.vararg:
                assigned.add(node.args.vararg.arg)
            if node.args.kwarg:
                assigned.add(node.args.kwarg.arg)
        if isinstance(node, ast.alias):
            assigned.add(node.asname or node.name.split(".")[0])
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.append((node.id, node.lineno))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            findings.append(Finding("Risk", "eval executes dynamically supplied text.", node.lineno, "Prefer a safe parser or explicit mapping."))
    builtins = set(__builtins__ if isinstance(__builtins__, dict) else dir(__builtins__))
    for name, line in loaded:
        if name not in assigned and name not in builtins:
            findings.append(Finding("Possible logic error", f"`{name}` may be used before it is defined.", line, f"Define `{name}` before this line or check its spelling."))
    if not findings:
        findings.append(Finding("No obvious issue", "The code is syntactically valid and passed the basic static checks."))
    return findings


def format_findings(findings: list[Finding]) -> str:
    sections = []
    for finding in findings:
        location = f" (line {finding.line})" if finding.line else ""
        text = f"**{finding.kind}{location}:** {finding.message}"
        if finding.hint:
            text += f"\n\n**How to fix:** {finding.hint}"
        sections.append(text)
    return "\n\n---\n\n".join(sections)


def build_prompt(code: str) -> str:
    return f"### Instruction:\nFix the bug in this Python code. Explain the cause and corrected approach.\n\n### Code:\n```python\n{code}\n```\n\n### Response:\n"


def model_explanation(code: str) -> str:
    model_path = os.getenv("DEBUGGER_MODEL_PATH", "outputs/code-debugger")
    if not os.path.isdir(model_path):
        return ""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        inputs = tokenizer(build_prompt(code), return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        answer = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return re.sub(r"^### Response:\s*", "", answer)
    except Exception as error:
        return f"Model explanation unavailable: {error}"


def debug_code(code: str) -> str:
    findings = format_findings(static_findings(code))
    explanation = model_explanation(code)
    if explanation:
        return f"{findings}\n\n## Fine-tuned model\n{explanation}"
    return findings + "\n\n_Model mode is off. Set `DEBUGGER_MODEL_PATH` to a trained checkpoint to add model-generated explanations._"


def clear_all() -> tuple[str, str]:
    return "", ""


def build_demo():
    import gradio as gr

    with gr.Blocks(title="PyExplain") as demo:
        gr.Markdown("# PyExplain\nPaste Python code to detect syntax and likely logic errors.")
        with gr.Row():
            code = gr.Code(label="Python code", language="python", lines=18)
            result = gr.Markdown(label="Diagnosis")
        with gr.Row():
            debug = gr.Button("Debug", variant="primary")
            clear = gr.Button("Clear")
        debug.click(debug_code, inputs=code, outputs=result)
        clear.click(clear_all, outputs=[code, result])
    return demo


if __name__ == "__main__":
    import gradio as gr

    build_demo().launch(theme=gr.themes.Soft())
