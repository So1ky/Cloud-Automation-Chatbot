"""Terraform(HCL) 파일을 최상위 블록 단위로 파싱하고, 편집 목록(replace/add/delete)을 기계적으로 적용한다.

개발 에이전트가 검증 피드백으로 재시도할 때 파일 전체를 다시 출력하면 리소스 누락·이름 변경이
생기므로, 변경할 블록만 받아 적용하고 나머지 블록은 바이트 그대로 보존한다.

주소 형식: resource.<type>.<name> / data.<type>.<name> / variable.<name> / output.<name> /
          module.<name> / provider.<name> / locals / terraform
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_HEADER_RE = re.compile(r'^\s*(\w+)((?:\s+"[^"]*")*)\s*$')
_HEREDOC_RE = re.compile(r"<<-?\s*(\w+)\r?\n")


@dataclass
class Block:
    address: str
    start: int   # 헤더 줄 시작 오프셋
    end: int     # 닫는 중괄호 다음 오프셋


def _skip_string(text: str, i: int) -> int:
    """i가 여는 따옴표 위치. 닫는 따옴표 다음 위치를 돌려준다. ${ } 보간 안의 중괄호·따옴표를 건너뛴다."""
    n = len(text)
    i += 1
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        if text.startswith("${", i):
            depth = 1
            i += 2
            while i < n and depth:
                if text[i] == '"':
                    i = _skip_string(text, i)
                    continue
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            continue
        i += 1
    return n


def parse_blocks(text: str) -> List[Block]:
    """최상위 블록 목록. 문자열·히어독·주석 안의 중괄호는 무시한다."""
    blocks: List[Block] = []
    n = len(text)
    i = 0
    depth = 0
    header_start = -1
    brace_pos = -1
    while i < n:
        c = text[i]
        if c == "#" or text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
            continue
        if text.startswith("/*", i):
            close = text.find("*/", i + 2)
            i = n if close == -1 else close + 2
            continue
        if c == '"':
            i = _skip_string(text, i)
            continue
        m = _HEREDOC_RE.match(text, i)
        if m:
            marker = m.group(1)
            j = m.end()
            while j < n:
                nl = text.find("\n", j)
                line = text[j:] if nl == -1 else text[j:nl]
                j = n if nl == -1 else nl + 1
                if line.strip() == marker:
                    break
            i = j
            continue
        if c == "{":
            if depth == 0:
                header_start = text.rfind("\n", 0, i) + 1
                brace_pos = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and header_start >= 0:
                hm = _HEADER_RE.match(text[header_start:brace_pos])
                if hm:
                    labels = re.findall(r'"([^"]*)"', hm.group(2))
                    address = ".".join([hm.group(1)] + labels)
                    blocks.append(Block(address, header_start, i + 1))
                header_start = -1
        i += 1
    return blocks


def _find(blocks: List[Block], address: str) -> Block | None:
    return next((b for b in blocks if b.address == address), None)


def _insert_text(text: str, block: Block, new_hcl: str) -> str:
    return text[: block.start] + new_hcl.strip("\n") + text[block.end:]


def _delete_text(text: str, block: Block) -> str:
    end = block.end
    while end < len(text) and text[end] == "\n":
        end += 1
    return text[: block.start] + text[end:]


def apply_patch(files: Dict[str, str], edits: List[dict]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """편집 목록을 적용한다. (새 파일들, 적용된 설명, 건너뛴 이유) 를 돌려준다.

    edit: {"file": "main.tf", "action": "replace|add|delete", "address": "resource.aws_lb.api", "hcl": "..."}
    - replace 대상이 없으면 add로, add 대상이 이미 있으면 replace로 관대하게 처리한다.
    - hcl 본문의 블록 주소가 address와 다르면 건너뛴다 (모델이 엉뚱한 블록을 보낸 경우).
    """
    files = dict(files)
    applied: List[str] = []
    skipped: List[str] = []
    for e in edits:
        file, action, address = e.get("file"), e.get("action"), (e.get("address") or "").strip()
        hcl = e.get("hcl") or ""
        if file not in files:
            skipped.append(f"{address}: 알 수 없는 파일 {file}")
            continue
        text = files[file]
        blocks = parse_blocks(text)
        target = _find(blocks, address)
        if action == "delete":
            if target is None:
                skipped.append(f"{address}: 삭제 대상 없음")
                continue
            files[file] = _delete_text(text, target)
            applied.append(f"delete {address}")
            continue
        if action not in ("replace", "add"):
            skipped.append(f"{address}: 알 수 없는 action {action}")
            continue
        new_blocks = parse_blocks(hcl)
        if len(new_blocks) != 1 or new_blocks[0].address != address:
            got = [b.address for b in new_blocks]
            skipped.append(f"{address}: hcl 블록 주소 불일치 {got}")
            continue
        if target is None:
            sep = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
            files[file] = text + sep + hcl.strip("\n") + "\n"
            applied.append(f"add {address}")
        else:
            files[file] = _insert_text(text, target, hcl)
            applied.append(f"replace {address}")
    return files, applied, skipped
