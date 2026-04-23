"""LLADA2/DMax-specific GSM8K prompt + scoring utilities."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Any, Iterable, Optional

from diffulex_bench.tasks.gsm8k.sdar_utils import ground_truth_string_from_doc
from diffulex_bench.tasks.utils import is_equal
from diffulex_bench.tasks.utils.math_utils import get_final_answer

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def doc_to_text_math_dmax_chat(doc: dict) -> str:
    """Format GSM8K prompt to match DMax generate_spd chat template."""
    question = doc["question"]
    user_content = f"{question}\nLet's think step by step\n"
    return (
        "<role>SYSTEM</role>detailed thinking off<|role_end|>"
        f"<role>HUMAN</role>{user_content}<|role_end|>"
        "<role>ASSISTANT</role>"
    )


def strip_wrappers(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("**", "").replace("__", "").strip("`")
    cleaned = cleaned.replace("\\boxed{", "").replace("\\fbox{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("$", "").replace("\\$", "")
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace(",", "")
    return cleaned.strip()


def canonicalize_numeric(candidate: str) -> Optional[str]:
    cleaned = strip_wrappers(candidate)
    cleaned = cleaned.rstrip(".。!！?？,，;；:：")
    cleaned = cleaned.replace("%", "")
    cleaned = cleaned.strip()
    if not cleaned:
        return None

    if re.fullmatch(r"-?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        if denominator == "0":
            return None
        value = Fraction(int(numerator), int(denominator))
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    fraction_value = Fraction(value)
    return str(fraction_value.numerator) if fraction_value.denominator == 1 else f"{fraction_value.numerator}/{fraction_value.denominator}"


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def extract_last_number(text: str) -> Optional[str]:
    matches = NUMBER_RE.findall(strip_wrappers(text))
    if not matches:
        return None
    return matches[-1]


def extract_answer_spans(text: str) -> list[str]:
    patterns = [
        r"(?is)####\s*([^\n]+)",
        r"(?is)<answer>\s*(.*?)\s*</answer>",
        r"(?is)Final Answer\s*[:：]\s*(.*?)(?=\n\s*\n|$)",
        r"(?is)The final answer is\s*(.*?)(?:\.?\s*I hope it is correct\.?|$)",
        r"(?im)^\s*Answer\s*[:：]\s*(.+?)\s*$",
    ]
    spans: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                for piece in match:
                    if piece and piece.strip():
                        spans.append(piece.strip())
            elif match and match.strip():
                spans.append(match.strip())
    return spans


def extract_boxed_contents(text: str) -> list[str]:
    matches: list[str] = []
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break
            cursor = idx + len(command)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                start = cursor + 1
                continue
            depth = 0
            content: list[str] = []
            end_idx = None
            for pos in range(cursor, len(text)):
                char = text[pos]
                if char == "{":
                    depth += 1
                    if depth > 1:
                        content.append(char)
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = pos
                        break
                    content.append(char)
                else:
                    content.append(char)
            if end_idx is not None:
                matches.append("".join(content).strip())
                start = end_idx + 1
            else:
                start = cursor + 1
    return matches


def extract_ground_truth_answer_candidates(doc: dict[str, Any]) -> list[str]:
    answer_text = ground_truth_string_from_doc(doc)
    if not answer_text:
        return []

    candidates: list[str] = []
    explicit_spans = extract_answer_spans(answer_text)
    if explicit_spans:
        explicit_number = extract_last_number(explicit_spans[-1])
        if explicit_number:
            candidates.append(explicit_number)

    last_number = extract_last_number(answer_text)
    if last_number:
        candidates.append(last_number)

    normalized = [canonicalize_numeric(candidate) for candidate in candidates]
    return dedupe_keep_order([item for item in normalized if item])


def extract_llm_final_answer_candidates(text: str) -> list[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    candidates: list[str] = []

    explicit_spans = extract_answer_spans(raw_text)
    for span in reversed(explicit_spans[-3:]):
        number = extract_last_number(span)
        if number:
            candidates.append(number)

    boxed = extract_boxed_contents(raw_text)
    if boxed:
        boxed_number = extract_last_number(boxed[-1])
        if boxed_number:
            candidates.append(boxed_number)

    tail_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in reversed(tail_lines[-5:]):
        number = extract_last_number(line)
        if number:
            candidates.append(number)

    tail_text = "\n".join(tail_lines[-8:]) if tail_lines else raw_text[-500:]
    tail_number = extract_last_number(tail_text)
    if tail_number:
        candidates.append(tail_number)

    normalized = [canonicalize_numeric(candidate) for candidate in candidates]
    return dedupe_keep_order([item for item in normalized if item])


def candidates_match(gold_candidates: list[str], pred_candidates: list[str]) -> bool:
    top_preds = pred_candidates[:2]
    if not top_preds:
        return False
    for gold in gold_candidates:
        for pred in top_preds:
            if gold == pred:
                return True
    return False


def process_results_math_dmax_chat(doc: dict, results: list[str]) -> dict[str, Any]:
    """dInfer-style numeric candidate matching for DMax chat outputs."""
    prediction = results[0] if results else ""
    pred_candidates = extract_llm_final_answer_candidates(prediction)
    gold_candidates = extract_ground_truth_answer_candidates(doc)

    if pred_candidates and gold_candidates:
        correct = candidates_match(gold_candidates, pred_candidates)
    else:
        extracted = get_final_answer(prediction)
        ground_truth = ground_truth_string_from_doc(doc)
        executor = ThreadPoolExecutor(max_workers=1)

        async def check():
            return await is_equal(extracted, ground_truth, executor)

        correct = asyncio.run(check())

    return {"exact_match": int(correct)}
