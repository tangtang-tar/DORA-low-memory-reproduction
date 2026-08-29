"""C1 的可靠答案提取与错误分类。

原则：优先读取明确的最终答案区域；截断文本不使用“最后一个数字”兜底。
"""

import re

from sal.utils.qwen_math_parser import math_equal, strip_string


FINAL_MARKERS = ["therefore", "thus", "hence", "final answer", "the answer is"]
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def balanced_boxed(text):
    """返回所有完整的 \boxed{...} 内容；不接受缺右花括号的截断 box。"""
    answers = []
    start = 0
    marker = "\\boxed{"
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        depth = 1
        cursor = index + len(marker)
        content_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append(text[content_start : cursor - 1])
        start = index + len(marker)
    return answers


def replace_complete_boxes(text):
    result = text
    for answer in balanced_boxed(text):
        result = result.replace(f"\\boxed{{{answer}}}", answer)
    return result


def math_spans(text):
    spans = []
    for pattern in [r"\\\[(.*?)\\\]", r"\$\$(.*?)\$\$", r"\\\((.*?)\\\)", r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)"]:
        spans.extend(
            (match.start(), match.group(1))
            for match in re.finditer(pattern, text, re.DOTALL)
        )
    return [content for _, content in sorted(spans)]


def clean_answer(answer):
    answer = answer.strip().strip("$ ")
    answer = answer.replace("\\left", "").replace("\\right", "")
    if "=" in answer:
        answer = answer.rsplit("=", 1)[-1].strip()
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(
        r"\s*(?:inches?|units?|degrees?|radians?|meters?|seconds?)\s*$",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = answer.strip().rstrip(".,;:")
    return answer


def expression_from_tail(tail):
    unboxed = replace_complete_boxes(tail)
    spans = math_spans(unboxed)
    if spans:
        return clean_answer(spans[-1])

    name_match = re.search(
        r"\b([A-Z][a-z]+)\b\s+(?:is|has)\s+(?:the\s+)?(?:student\s+with\s+)?(?:the\s+)?greatest",
        unboxed,
        re.IGNORECASE,
    )
    if name_match:
        return name_match.group(1).title()

    number_match = re.search(
        r"(?:is|are|equals?|has)\s+\**\s*(-?\d+(?:\.\d+)?)\b",
        unboxed,
        re.IGNORECASE,
    )
    if number_match:
        return number_match.group(1)

    word_match = re.search(
        r"\b(?:only|is|are)\s+(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        unboxed,
        re.IGNORECASE,
    )
    if word_match:
        return NUMBER_WORDS[word_match.group(1).lower()]
    return None


def extract_reliable_answer(text, is_truncated=False):
    boxes = balanced_boxed(text)

    # 若 box 只包住坐标的一部分，优先读取包含它的完整最终数学表达式。
    marker_matches = [
        (match.start(), marker)
        for marker in FINAL_MARKERS
        for match in re.finditer(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE)
    ]
    marker_matches.append((-1, ""))
    final_position, final_marker = max(marker_matches)
    marker_is_usable = final_position >= 0 and (
        not is_truncated
        or (
            len(text) - final_position <= 300
            and (final_marker != "thus" or len(text) - final_position <= 100)
        )
    )
    if marker_is_usable:
        answer = expression_from_tail(text[final_position:])
        if answer:
            return {"answer": answer, "source": "explicit_final", "parse_success": True}

    if boxes:
        last_box = clean_answer(boxes[-1])
        tail = text[max(0, text.rfind("\n", 0, text.rfind("\\boxed{")) - 300) :]
        containing_expression = expression_from_tail(tail)
        answer = containing_expression or last_box
        return {"answer": answer, "source": "boxed", "parse_success": True}

    # 截断前若已经出现 Therefore/Thus，则允许读取其前一条完整等式。
    if is_truncated and marker_is_usable:
        conclusion_context = text[max(0, final_position - 600) :]
        unique_count = re.search(
            r"(?:only|there (?:is|are))\s+(zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:unique|distinct)",
            conclusion_context,
            re.IGNORECASE,
        )
        if unique_count:
            return {
                "answer": NUMBER_WORDS[unique_count.group(1).lower()],
                "source": "pre_truncation_conclusion",
                "parse_success": True,
            }
        spans = math_spans(replace_complete_boxes(text[:final_position]))
        if spans:
            return {
                "answer": clean_answer(spans[-1]),
                "source": "pre_truncation_result",
                "parse_success": True,
            }

    if not is_truncated:
        last_section = text[-1200:]
        answer = expression_from_tail(last_section)
        if answer:
            return {"answer": answer, "source": "concluding_context", "parse_success": True}

    return {"answer": None, "source": "no_explicit_answer", "parse_success": False}


def normalized_text(answer):
    if answer is None:
        return None
    answer = clean_answer(answer)
    answer = answer.replace("π", "\\pi").replace("θ", "\\theta")
    return re.sub(r"\s+", "", answer).lower()


def answers_equivalent(first, second):
    if first is None or second is None:
        return first is second
    if normalized_text(first) == normalized_text(second):
        return True
    try:
        return bool(
            math_equal(
                strip_string(normalized_text(first)),
                strip_string(normalized_text(second)),
            )
        )
    except Exception:
        return False
