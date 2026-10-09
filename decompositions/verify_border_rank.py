#!/usr/bin/env python3
"""Verify an exact .exp decomposition and a nullcone witness.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sympy import Matrix, Symbol, simplify

from load_exp_sympy import load_exp_as_matrix_triples
from verify_exp import diff_tensors, expand_decomposition, infer_dimensions, target_trace_tensor


T = Symbol("t", positive=True)


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def parse_basis(value: Any, name: str, size: int) -> Matrix:
    matrix = Matrix(value)
    require(matrix.shape == (size, size), f"basis {name} must be {size}x{size}")
    require(matrix.det() != 0, f"basis {name} is singular")
    return matrix


def exponent(entry: Any) -> int:
    coefficient, power = simplify(entry).as_coeff_exponent(T)
    require(not coefficient.has(T), f"not a Laurent monomial in t: {entry}")
    require(power.is_Integer, f"nonintegral exponent in {entry}")
    return int(power)


def matrix_order(matrix: Matrix) -> int:
    powers = [exponent(entry) for entry in matrix if simplify(entry) != 0]
    require(bool(powers), "deleted term has a zero factor")
    return min(powers)


def verify(exp_path: Path, witness_path: Path) -> tuple[tuple[int, int, int], int, int, list[int]]:
    tensor, rank = expand_decomposition(exp_path)
    dimensions = infer_dimensions(tensor)
    require(dimensions == tuple(sorted(dimensions)),
            f"dimensions are not in canonical nondecreasing order: {dimensions}")
    require(not diff_tensors(tensor, target_trace_tensor(*dimensions)),
            ".exp file is not an exact decomposition of Tr(ABC)")

    data = json.loads(witness_path.read_text())
    require(data.get("schema_version") == 1, "unsupported witness schema")
    deleted = data["deleted_terms"]
    require(isinstance(deleted, list) and deleted, "deleted_terms must be nonempty")
    require(len(deleted) == len(set(deleted)), "deleted_terms contains duplicates")
    require(all(isinstance(number, int) and 1 <= number <= rank for number in deleted),
            "deleted term number is out of range")

    p, q, r = dimensions
    sizes = {"A": p, "B": q, "C": r}
    bases = {name: parse_basis(data["basis_matrices"][name], name, size)
             for name, size in sizes.items()}
    weights: dict[str, list[int]] = {}
    for name, size in sizes.items():
        values = data["weights"][name]
        require(isinstance(values, list) and len(values) == size,
                f"weights {name} must have length {size}")
        require(all(isinstance(value, int) for value in values),
                f"weights {name} must be integers")
        weights[name] = values
    diagonals = {name: Matrix.diag(*(T ** value for value in values))
                 for name, values in weights.items()}

    triples = load_exp_as_matrix_triples(exp_path)
    orders: list[int] = []
    for number in deleted:
        u, v, w = triples[number - 1]
        transformed = (
            diagonals["A"] * bases["A"].inv() * u * bases["B"] * diagonals["B"].inv(),
            diagonals["B"] * bases["B"].inv() * v * bases["C"] * diagonals["C"].inv(),
            diagonals["C"] * bases["C"].inv() * w * bases["A"] * diagonals["A"].inv(),
        )
        order = sum(matrix_order(matrix) for matrix in transformed)
        require(order > 0, f"term {number} does not tend to zero (order {order})")
        orders.append(order)

    return dimensions, rank, rank - len(deleted), orders


def case_pairs(path: Path) -> list[tuple[Path, Path]]:
    if path.is_file():
        raise VerificationError("directory scan expects a directory")
    return [(witness.parent / "decomposition.exp", witness)
            for witness in sorted(path.rglob("witness.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_or_directory", type=Path)
    parser.add_argument("witness", nargs="?", type=Path)
    args = parser.parse_args()
    pairs = ([(args.exp_or_directory, args.witness)] if args.witness is not None
             else case_pairs(args.exp_or_directory))
    require(bool(pairs), "no witness.json files found")

    failures = 0
    for exp_path, witness_path in pairs:
        try:
            dimensions, rank, border, orders = verify(exp_path, witness_path)
            print(f"VALID   {exp_path.parent}: dims={dimensions}, rank={rank}, "
                  f"border<={border}, deleted-term orders={orders}")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures += 1
            print(f"INVALID {exp_path} + {witness_path}: {error}")
    print(f"\n{len(pairs) - failures}/{len(pairs)} witnesses verified")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
