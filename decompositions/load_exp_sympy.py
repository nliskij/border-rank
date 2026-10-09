from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sympy import Symbol, Rational
from sympy import Matrix, zeros, simplify

from verify_exp import (
    LinearForm,
    ParseError,
    Tensor,
    infer_dimensions,
    parse_basis_name,
    parse_rank_one_line,
)


def linear_form_to_matrix(form: LinearForm, prefix: str, rows: int, cols: int) -> Matrix:
    matrix = zeros(rows, cols)

    for name, coeff in form.items():
        name_prefix, row, col = parse_basis_name(name)
        if name_prefix != prefix:
            raise ParseError(f"expected only {prefix}-basis terms, got {name!r}")
        if not (1 <= row <= rows and 1 <= col <= cols):
            raise ParseError(
                f"basis term {name!r} is out of bounds for a {rows}x{cols} matrix"
            )
        matrix[row - 1, col - 1] = coeff

    return matrix


def load_exp_as_matrix_triples(path: Path) -> list[tuple[Matrix, Matrix, Matrix]]:
    triples: list[tuple[LinearForm, LinearForm, LinearForm]] = []
    dims_tensor: Tensor = defaultdict(int)

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        triple = parse_rank_one_line(stripped, line_number)
        triples.append(triple)

        a_form, b_form, c_form = triple
        for a_name, a_coeff in a_form.items():
            for b_name, b_coeff in b_form.items():
                for c_name, c_coeff in c_form.items():
                    dims_tensor[(a_name, b_name, c_name)] += a_coeff * b_coeff * c_coeff

    if not triples:
        raise ParseError("file contains no rank-one terms")

    p, q, r = infer_dimensions(dims_tensor)
    return [
        (
            linear_form_to_matrix(a_form, "a", p, q),
            linear_form_to_matrix(b_form, "b", q, r),
            linear_form_to_matrix(c_form, "c", r, p),
        )
        for a_form, b_form, c_form in triples
    ]


def verify_decomp(decomp: list[tuple[Matrix, Matrix, Matrix]]) -> bool:
    m, n = decomp[0][0].shape
    n, p = decomp[0][1].shape

    for i1 in range(m):
        for j1 in range(n):
            for i2 in range(n):
                for j2 in range(p):
                    for i3 in range(p):
                        for j3 in range(m):
                            a = zeros(m, n)
                            b = zeros(n, p)
                            c = zeros(p, m)
                            a[i1, j1] = 1
                            b[i2, j2] = 1
                            c[i3, j3] = 1
                            s = 0
                            for u, v, w in decomp:
                                s += (
                                    (a @ u.transpose()).trace()
                                    * (b @ v.transpose()).trace()
                                    * (c @ w.transpose()).trace()
                                )
                            if s != (i1 == j3) and (j1 == i2) and (j2 == i3):
                                print(f"bad eval at {(i1, j1), (i2, j2), (i3, j3)}")
                                return False
    return True


def sandwich(decomp, r, s, t):
    return [(r @ u @ s.inv(), s @ v @ t.inv(), t @ w @ r.inv()) for u, v, w in decomp]


if __name__ == "__main__":
    decomp = load_exp_as_matrix_triples(Path("m245-33-good.exp"))
    eps = Symbol("eps")
    r = Matrix([[1, 0], [0, eps]])
    s = Matrix([
        [1, 0, Rational(19, 9) * (eps - 1), 0],
        [0, eps, 0, 0],
        [0, 0, eps, 0],
        [0, 0, 0, 1],
    ])
    t = Matrix([
        [1, 0, 0, 0, 0],
        [0, eps, 0, 0, 0],
        [0, 0, eps, 0, 0],
        [0, eps - 1, eps - 1, 1, 0],
        [0, eps - 1, 2 * eps - 2, 0, 1],
    ])
    for i, (u, v, w) in enumerate(sandwich(decomp, r, s, t)):
        if not (i == 16 or i == 26):
            print((simplify(u), simplify(v), simplify(w)))
