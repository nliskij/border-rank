#!/usr/bin/env python3
"""Emit a witnessed border-rank decomposition as executable SymPy code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sympy import Matrix, Rational, Symbol, cancel, diag


# This script lives one directory below the verifier and .exp loader.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from load_exp_sympy import load_exp_as_matrix_triples  # noqa: E402
from verify_border_rank import VerificationError, verify  # noqa: E402


T = Symbol("t", nonzero=True)


def parse_rational_matrix(value: object) -> Matrix:
    return Matrix([[Rational(entry) for entry in row] for row in value])  # type: ignore[union-attr]


def transform_terms(
    triples: list[tuple[Matrix, Matrix, Matrix]], witness: dict[str, object]
) -> list[tuple[Matrix, Matrix, Matrix]]:
    bases = {
        name: parse_rational_matrix(witness["basis_matrices"][name])  # type: ignore[index]
        for name in ("A", "B", "C")
    }
    weights = witness["weights"]
    diagonals = {
        name: diag(*(T ** int(weight) for weight in weights[name]))  # type: ignore[index]
        for name in ("A", "B", "C")
    }

    transformed = []
    for u, v, w in triples:
        transformed.append((
            diagonals["A"] * bases["A"].inv() * u * bases["B"] * diagonals["B"].inv(),
            diagonals["B"] * bases["B"].inv() * v * bases["C"] * diagonals["C"].inv(),
            diagonals["C"] * bases["C"].inv() * w * bases["A"] * diagonals["A"].inv(),
        ))
    return [tuple(matrix.applyfunc(cancel) for matrix in triple) for triple in transformed]


def reduced_terms(
    triples: list[tuple[Matrix, Matrix, Matrix]], witness: dict[str, object]
) -> list[tuple[Matrix, Matrix, Matrix]]:
    transformed = transform_terms(triples, witness)
    deleted = witness.get("deleted_terms")
    if deleted is not None:
        omitted = set(deleted)  # type: ignore[arg-type]
        return [triple for number, triple in enumerate(transformed, 1) if number not in omitted]

    combination = witness["nullcone_combination"]  # type: ignore[index]
    scales = [Rational(1) for _ in transformed]
    for number, coefficient in zip(
        combination["terms"], combination["coefficients"]  # type: ignore[index]
    ):
        scales[int(number) - 1] -= Rational(coefficient)

    result = []
    for scale, (u, v, w) in zip(scales, transformed):
        if scale != 0:
            result.append(((scale * u).applyfunc(cancel), v, w))
    return result


def scalar_code(value: object) -> str:
    value = cancel(value)
    if value == 0:
        return "0"
    coefficient, power = value.as_coeff_exponent(T)
    if coefficient.has(T) or not power.is_Integer:
        raise ValueError(f"expected a rational Laurent monomial, got {value}")

    numerator, denominator = map(int, coefficient.as_numer_denom())
    if denominator == 1:
        coefficient_code = str(numerator)
    else:
        coefficient_code = f"sp.Rational({numerator}, {denominator})"
    exponent = int(power)
    if exponent == 0:
        return coefficient_code
    power_code = "t" if exponent == 1 else f"t**({exponent})"
    if coefficient == 1:
        return power_code
    if coefficient == -1:
        return "-" + power_code
    return coefficient_code + "*" + power_code


def matrix_code(matrix: Matrix, indentation: str) -> list[str]:
    lines = [indentation + "sp.Matrix(["]
    for row in matrix.tolist():
        lines.append(indentation + "    [" + ", ".join(map(scalar_code, row)) + "],")
    lines.append(indentation + "])")
    return lines


def sympy_program(
    case: Path, terms: list[tuple[Matrix, Matrix, Matrix]],
    dimensions: tuple[int, int, int], original_rank: int, border_rank: int,
) -> str:
    p, q, r = dimensions
    lines = [
        "import sympy as sp",
        "",
        "t = sp.Symbol('t')",
        "",
        f"# Rank-{border_rank} border-rank family from rank-{original_rank} case {case.name}.",
        "decomp = [",
    ]
    for triple in terms:
        lines.append("    (")
        for matrix in triple:
            lines.extend(matrix_code(matrix, "        "))
            lines[-1] += ","
        lines.append("    ),")
    lines.extend([
        "]",
        "",
        "",
        "error_tests = []",
        f"for i1 in range({p}):",
        f"    for j1 in range({q}):",
        f"        for i2 in range({q}):",
        f"            for j2 in range({r}):",
        f"                for i3 in range({r}):",
        f"                    for j3 in range({p}):",
        f"                        a = sp.zeros({p}, {q})",
        f"                        b = sp.zeros({q}, {r})",
        f"                        c = sp.zeros({r}, {p})",
        "                        a[i1, j1] = 1",
        "                        b[i2, j2] = 1",
        "                        c[i3, j3] = 1",
        "                        correct_val = (",
        "                            1 if j1 == i2 and j2 == i3 and j3 == i1 else 0",
        "                        )",
        "                        s = 0",
        "                        for u, v, w in decomp:",
        "                            s += (",
        "                                (a @ u.transpose()).trace()",
        "                                * (b @ v.transpose()).trace()",
        "                                * (c @ w.transpose()).trace()",
        "                            )",
        "                        err = sp.simplify(correct_val - s)",
        "                        print(err)",
        "                        coeff, power = err.leadterm(t)",
        "                        good = coeff == 0 or power >= 1",
        "                        if not good:",
        "                            print('term without O(t) error!')",
        "                        error_tests.append(good)",
        "",
        "print('all tests passed: ', all(error_tests))",
        "print('length of decomp is', len(decomp))",
        "print('dimensions are ', decomp[0][0].shape, decomp[0][1].shape, decomp[0][2].shape)",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", type=Path,
        help="case directory containing decomposition.exp and witness.json",
    )
    parser.add_argument("-o", "--output", type=Path, help="write code here instead of stdout")
    args = parser.parse_args()

    case = args.case.resolve()
    exp_path = case / "decomposition.exp"
    witness_path = case / "witness.json"
    verification = verify(exp_path, witness_path)
    # Only the first three fields are part of the helper's interface.  Older
    # verifier versions returned four fields; current versions additionally
    # report the witness kind.
    dimensions, original_rank, border_rank = verification[:3]
    witness = json.loads(witness_path.read_text())
    terms = reduced_terms(load_exp_as_matrix_triples(exp_path), witness)
    if len(terms) != border_rank:
        raise VerificationError(
            f"constructed {len(terms)} terms, but the witness proves rank {border_rank}"
        )
    program = sympy_program(case, terms, dimensions, original_rank, border_rank)
    if args.output is None:
        sys.stdout.write(program)
    else:
        args.output.write_text(program)
        print(
            f"wrote rank-{border_rank} family for dimensions {dimensions} to {args.output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
