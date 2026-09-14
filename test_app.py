import ast

from app import static_findings


def test_valid_code():
    findings = static_findings("print('hello')")
    assert findings[0].kind == "No obvious issue"


def test_syntax_error():
    findings = static_findings("for item in items\n    print(item)")
    assert findings[0].kind == "Syntax error"
    assert findings[0].line == 1


def test_possible_undefined_name():
    findings = static_findings("print(total)")
    assert findings[0].kind == "Possible logic error"
    assert "total" in findings[0].message
