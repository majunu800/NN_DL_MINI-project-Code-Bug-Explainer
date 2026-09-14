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
    correction: str = ""


def static_findings(code: str) -> list[Finding]:
    if not code.strip():
        return [Finding("Input", "Paste Python code into the editor.")]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        line = error.lineno
        detail = error.msg
        hint = "A syntax error stops the program before it can run. Check punctuation, indentation, brackets, and spelling."
        correction = "Check the marked line and add the missing punctuation or indentation."
        if "expected ':'" in detail:
            correction = "Add a colon (`:`) at the end of the statement, for example: `for item in items:`."
        return [Finding("Syntax error", f"Python could not understand this line: {detail}.", line, hint, correction)]

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
            findings.append(Finding(
                "Safety warning",
                "`eval()` runs text as Python code, so user input could run unexpected commands.",
                node.lineno,
                "This can create a security problem when the text comes from outside the program.",
                "Use `ast.literal_eval()` when you only need to read safe Python values.",
            ))
    builtins = set(__builtins__ if isinstance(__builtins__, dict) else dir(__builtins__))
    for name, line in loaded:
        if name not in assigned and name not in builtins:
            findings.append(Finding(
                "Possible logic error",
                f"The name `{name}` is used here, but Python cannot find where it was created.",
                line,
                "The program may stop with a NameError, or the variable name may be misspelled.",
                f"Check the spelling and create `{name}` before line {line}.",
            ))
    if not findings:
        findings.append(Finding(
            "Looks good",
            "Python can read this code, and the basic checks did not find an obvious problem.",
            hint="Run the program with real examples too. Static checks cannot prove that every result is correct.",
        ))
    return findings


def format_findings(findings: list[Finding]) -> str:
    sections = ["## Diagnosis\n\nHere is the result in simple English:"]
    for finding in findings:
        location = f" (line {finding.line})" if finding.line else ""
        text = f"### {finding.kind}{location}\n\n**What happened:** {finding.message}"
        if finding.hint:
            text += f"\n\n**Why it matters:** {finding.hint}"
        if finding.correction:
            text += f"\n\n**Recommended fix:** {finding.correction}"
        sections.append(text)
    return "\n\n---\n\n".join(sections)


def build_prompt(code: str) -> str:
    return f"### Instruction:\nFix the bug in this Python code. Explain the cause and corrected approach in simple English.\n\n### Code:\n```python\n{code}\n```\n\n### Response:\n"


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
        return f"{findings}\n\n---\n\n## AI explanation\n\n{explanation}"
    return findings + "\n\n---\n\n*Static checker active. A trained model can add a longer AI explanation when `DEBUGGER_MODEL_PATH` is configured.*"


def clear_all() -> tuple[str, str]:
    return "", ""


def build_demo():
    import gradio as gr

    with gr.Blocks(title="PyExplain | Python Bug Explainer") as demo:
        gr.Markdown(
            "# PyExplain\n### Understand your Python bug in simple English\nPaste code, press **Debug**, and get the problem, reason, and recommended fix.",
            elem_classes="hero",
        )
        with gr.Row():
            with gr.Column(elem_classes="panel"):
                code = gr.Code(
                    label="1. Paste Python code",
                    language="python",
                    lines=18,
                    value="def average(values):\n    return sum(values) / len(value)",
                )
                gr.Markdown("**Try an example:** missing colon, undefined variable, or valid code.")
                with gr.Row():
                    debug = gr.Button("Debug code", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(elem_classes=["panel", "result"]):
                result = gr.Markdown("### 2. Your explanation\n\nPress **Debug code** to see a clear diagnosis.")
        debug.click(debug_code, inputs=code, outputs=result)
        clear.click(clear_all, outputs=[code, result])
    return demo


if __name__ == "__main__":
    import gradio as gr

    build_demo().launch(theme=gr.themes.Soft(), css="""
    .gradio-container { background: linear-gradient(135deg, #f5f7ff 0%, #fff8f1 100%); }
    .hero { padding: 24px 28px; border-radius: 18px; background: linear-gradient(120deg, #243b8f, #4169c8); color: white; margin-bottom: 16px; }
    .hero h1 { color: white !important; margin-bottom: 6px; }
    .panel { border: 1px solid #dfe5f2; border-radius: 14px; background: rgba(255,255,255,.86); padding: 8px; }
    .result { border-left: 5px solid #f28c28; }
    footer { display: none !important; }
    """)
