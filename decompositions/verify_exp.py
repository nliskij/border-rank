#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

from sympy import Rational


BasisTerm = tuple[str, str, str]
LinearForm = dict[str, Rational]
Tensor = DefaultDict[BasisTerm, Rational]

LINE_RE = re.compile(r"^\(([^()]*)\)\*\(([^()]*)\)\*\(([^()]*)\)$")
TOKEN_RE = re.compile(r"[+-]?[^+-]+")
VAR_RE = re.compile(r"^(?:(\d+(?:/\d+)?))?([abc]\d\d)$")


class ParseError(ValueError):
    pass


def parse_linear_form(text: str) -> LinearForm:
    form: LinearForm = defaultdict(int)
    compact = text.replace(" ", "")
    if not compact:
        raise ParseError("empty linear form")

    for token in TOKEN_RE.findall(compact):
        if token in {"+", "-"}:
            raise ParseError(f"invalid token {token!r}")
        sign = 1
        if token[0] == "+":
            token = token[1:]
        elif token[0] == "-":
            sign = -1
            token = token[1:]
        match = VAR_RE.fullmatch(token)
        if not match:
            raise ParseError(f"invalid basis term {token!r}")
        coeff_text, variable = match.groups()
        coeff = Rational(coeff_text) if coeff_text else Rational(1)
        form[variable] += sign * coeff

    if not form:
        raise ParseError("linear form has no terms")
    return dict(form)


def parse_rank_one_line(line: str, line_number: int) -> tuple[LinearForm, LinearForm, LinearForm]:
    stripped = line.strip()
    match = LINE_RE.fullmatch(stripped)
    if not match:
        raise ParseError(f"line {line_number}: expected '(...) *(...) *(...)' product")

    a_form = parse_linear_form(match.group(1))
    b_form = parse_linear_form(match.group(2))
    c_form = parse_linear_form(match.group(3))

    if any(not name.startswith("a") for name in a_form):
        raise ParseError(f"line {line_number}: first factor must only contain a-basis terms")
    if any(not name.startswith("b") for name in b_form):
        raise ParseError(f"line {line_number}: second factor must only contain b-basis terms")
    if any(not name.startswith("c") for name in c_form):
        raise ParseError(f"line {line_number}: third factor must only contain c-basis terms")

    return a_form, b_form, c_form


def expand_decomposition(path: Path) -> tuple[Tensor, int]:
    tensor: Tensor = defaultdict(int)
    rank = 0

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        a_form, b_form, c_form = parse_rank_one_line(stripped, line_number)
        rank += 1
        for a_name, a_coeff in a_form.items():
            for b_name, b_coeff in b_form.items():
                for c_name, c_coeff in c_form.items():
                    tensor[(a_name, b_name, c_name)] += a_coeff * b_coeff * c_coeff

    if rank == 0:
        raise ParseError("file contains no rank-one terms")

    return tensor, rank


def parse_basis_name(name: str) -> tuple[str, int, int]:
    if len(name) != 3 or name[0] not in "abc" or not name[1:].isdigit():
        raise ParseError(f"invalid basis name {name!r}")
    return name[0], int(name[1]), int(name[2])


def infer_dimensions(tensor: Tensor) -> tuple[int, int, int]:
    a_rows: set[int] = set()
    a_cols: set[int] = set()
    b_rows: set[int] = set()
    b_cols: set[int] = set()
    c_rows: set[int] = set()
    c_cols: set[int] = set()

    for a_name, b_name, c_name in tensor:
        a_prefix, a_row, a_col = parse_basis_name(a_name)
        b_prefix, b_row, b_col = parse_basis_name(b_name)
        c_prefix, c_row, c_col = parse_basis_name(c_name)
        if (a_prefix, b_prefix, c_prefix) != ("a", "b", "c"):
            raise ParseError(f"invalid basis triple {(a_name, b_name, c_name)!r}")
        a_rows.add(a_row)
        a_cols.add(a_col)
        b_rows.add(b_row)
        b_cols.add(b_col)
        c_rows.add(c_row)
        c_cols.add(c_col)

    p = max(a_rows)
    q = max(a_cols)
    r = max(b_cols)

    if a_rows != set(range(1, p + 1)):
        raise ParseError("a-basis row indices are not contiguous from 1")
    if a_cols != set(range(1, q + 1)):
        raise ParseError("a-basis column indices are not contiguous from 1")
    if b_rows != set(range(1, q + 1)):
        raise ParseError("b-basis row indices do not match a-basis column indices")
    if b_cols != set(range(1, r + 1)):
        raise ParseError("b-basis column indices are not contiguous from 1")
    if c_rows != set(range(1, r + 1)):
        raise ParseError("c-basis row indices do not match b-basis column indices")
    if c_cols != set(range(1, p + 1)):
        raise ParseError("c-basis column indices do not match a-basis row indices")

    return p, q, r


def target_trace_tensor(p: int, q: int, r: int) -> Tensor:
    tensor: Tensor = defaultdict(int)
    for i in range(1, p + 1):
        for j in range(1, q + 1):
            for k in range(1, r + 1):
                tensor[(f"a{i}{j}", f"b{j}{k}", f"c{k}{i}")] += 1
    return tensor


def diff_tensors(actual: Tensor, expected: Tensor) -> list[tuple[BasisTerm, int, int]]:
    keys = set(actual) | set(expected)
    diffs = []
    for key in sorted(keys):
        actual_value = actual.get(key, 0)
        expected_value = expected.get(key, 0)
        if actual_value != expected_value:
            diffs.append((key, actual_value, expected_value))
    return diffs


def iter_exp_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(file_path for file_path in path.rglob("*.exp") if file_path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that .exp decompositions equal the 3x3 matrix multiplication tensor Tr(ABC)."
    )
    parser.add_argument("path", type=Path, help="An .exp file or a directory containing .exp files")
    parser.add_argument(
        "--show-diff",
        type=int,
        default=10,
        help="How many coefficient mismatches to show for invalid files (default: 10)",
    )
    args = parser.parse_args()

    files = iter_exp_files(args.path)
    if not files:
        print(f"No .exp files found under {args.path}", file=sys.stderr)
        return 1

    failures = 0

    for file_path in files:
        try:
            actual, rank = expand_decomposition(file_path)
            dims = infer_dimensions(actual)
            expected = target_trace_tensor(*dims)
            diffs = diff_tensors(actual, expected)
        except ParseError as exc:
            failures += 1
            print(f"INVALID {file_path}: parse error: {exc}")
            continue

        if diffs:
            failures += 1
            print(f"INVALID {file_path}: dims={dims}, rank={rank}, mismatches={len(diffs)}")
            for key, actual_value, expected_value in diffs[: args.show_diff]:
                print(f"  {key}: actual={actual_value}, expected={expected_value}")
        else:
            print(f"VALID   {file_path}: dims={dims}, rank={rank}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
