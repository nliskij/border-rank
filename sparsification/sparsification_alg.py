"""
Symbolic version of the sparsification algorithm with Laurent-polynomial phi entries.

Main changes:
- Matrix entries may be numbers or SymPy Laurent expressions in epsilon.
- Phi entries are no longer restricted to numeric scalars.  Each phi entry is
  itself a Laurent polynomial with Z3 coefficient variables.
- The allowed powers in phi are automatically inferred from the input matrix
  using the dimension-dependent Laurent bound discussed in the proof:

      alpha = min(0, matrix minimum power)
      beta  = max(0, matrix maximum power)
      d = beta - alpha
      s = number of columns of A = length of each phi column
      phi supports powers from -beta - (s - 1)d to -alpha + (s - 1)d.

  Example: if matrix entries use epsilon powers from -2 to 4 and A has 9 columns,
  then d=6 and every phi entry is allowed to use epsilon powers
  -4 - 8*6, ..., 2 + 8*6, i.e. -52, ..., 50.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import importlib.util
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sympy as sp
import z3


# -----------------------------
# User-tunable settings
# -----------------------------


"""
PLEASE READ:

Look at the tunables here and change it to what you need. 

The MATRIX_FILE should be pointed towards a python file that contains the decomposition. 

The BASE_DIR should be pointed towards the location in which you want to save the files.

<algorithmFolder>/<name>/Optimal is the resulting folder where the new design matrices
and the phi_inv matrices will be saved. 

SYMBOLIC_VARIABLE_SCORE can be changed to make epsilons weigh more/less.

SYMBOLIC_SCORE_MODE changes how the code counts the score. If it is "variable," it will 
check if it contains an epsilon to add to the score. If it is "term," it will check how
many epsilon terms exist. For example, e^-1 + e^4 will be counted to have a score of 
SYMBOLIC_VARIABLE_SCORE + base_arithmetic_score if it is set to "variable," whereas it 
will be scored as 2*SYMBOLIC_VARIABLE_SCORE + base_arithmetic_score if it is set to 
"term."
"""
name = "3_3_3"
algorithmFolder = "APA_algorithms"
array_name = "u"  # "u", "v", or "w"

USE_SMIRNOV_SYMBOLIC_INPUT = True
MATRIX_FILE = Path(__file__).with_name("smirnov_333_20_encoding_literals.py")

SYMBOLIC_VARIABLE_SCORE = 5
SYMBOLIC_SCORE_MODE = "term"  # "variable" or "term"

SAVE_OUTPUTS = True

BASE_DIR = Path("path/to/algorithms")

# -----------------------------
# Loading helpers
# -----------------------------

def load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_factor_matrices(mats: Sequence[sp.Matrix]) -> np.ndarray:
    """Convert factor matrices into the paper orientation t x nm.

    For Smirnov 3x3x3 with t=20, each 3x3 factor matrix becomes one row,
    so U/V/W become 20 x 9.
    """
    rows = []
    for M in mats:
        M = sp.Matrix(M)
        rows.append([sp.expand(M[i, j]) for i in range(M.rows) for j in range(M.cols)])
    return np.array(rows, dtype=object)


def load_input_matrix() -> np.ndarray:
    if USE_SMIRNOV_SYMBOLIC_INPUT:
        module = load_python_module(MATRIX_FILE)
        if array_name == "u":
            return flatten_factor_matrices(module.U)
        if array_name == "v":
            return flatten_factor_matrices(module.V)
        if array_name == "w":
            return flatten_factor_matrices(module.W)
        raise ValueError("array_name must be 'u', 'v', or 'w'")

    folder = BASE_DIR / algorithmFolder / name
    if algorithmFolder == "DeepMindAlgorithms":
        return np.loadtxt(folder / f"{name}_{array_name}_arr.csv", dtype=object, delimiter=",")

    if algorithmFolder == "FlipGraphAlgorithms":
        alg_path = folder / f"flips_mod0_{name}.alg"
        design_matrices = []
        temp = []
        with open(alg_path) as alg:
            for line_raw in alg:
                line = line_raw.rstrip().split(" ")
                if line[0] == "#":
                    design_matrices.append(temp.copy())
                    temp = []
                else:
                    temp.append([sp.Rational(x) for x in line])
        design_matrices.append(temp.copy())
        offset = {"u": 1, "v": 2, "w": 3}[array_name]
        return np.array(design_matrices[offset], dtype=object).T

    raise ValueError(f"Unknown algorithmFolder: {algorithmFolder}")


# -----------------------------
# SymPy/Z3 conversion helpers
# -----------------------------

MonomialKey = Tuple[int, ...]
PolyCoeffs = Dict[MonomialKey, sp.Rational]
Z3PolyCoeffs = Dict[MonomialKey, z3.ArithRef]


def infer_symbolic_variables(matrix: np.ndarray) -> Tuple[sp.Symbol, ...]:
    symbols = set()
    for x in matrix.flat:
        symbols |= sp.sympify(x).free_symbols
    return tuple(sorted(symbols, key=lambda s: s.name))


def z3_real(value) -> z3.ArithRef:
    q = sp.Rational(value)
    if q.q == 1:
        return z3.RealVal(str(q.p))
    return z3.RealVal(f"{q.p}/{q.q}")


def z3_value_to_fraction(value) -> Fraction:
    value = z3.simplify(value)
    if z3.is_int_value(value):
        return Fraction(value.as_long(), 1)
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise ValueError(f"Expected rational Z3 value, got {value}")


def frac_to_sympy(x: Fraction) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator)


def expression_to_laurent_coeffs(expr, symbols: Sequence[sp.Symbol]) -> PolyCoeffs:
    """Convert an expression into {monomial_power_tuple: rational coefficient}.

    For symbols=(epsilon,):
        1 - epsilon**3 -> {(0,): 1, (3,): -1}
        epsilon**-2    -> {(-2,): 1}
    """
    expr = sp.expand(sp.sympify(expr))
    if expr == 0:
        return {}

    coeffs: PolyCoeffs = {}
    for term in sp.Add.make_args(expr):
        powers = term.as_powers_dict()
        key = []
        monomial = sp.Integer(1)
        for sym in symbols:
            power = powers.get(sym, sp.Integer(0))
            if not power.is_Integer:
                raise ValueError(f"Non-integer Laurent power {power} in term {term}")
            power_int = int(power)
            key.append(power_int)
            monomial *= sym ** power_int

        coeff = sp.simplify(term / monomial)
        if coeff.free_symbols:
            raise ValueError(
                f"Coefficient {coeff} still contains symbols {coeff.free_symbols}. "
                "Add those symbols to the symbolic variable set or specialize them."
            )
        key_tuple = tuple(key)
        coeffs[key_tuple] = sp.Rational(coeffs.get(key_tuple, 0) + coeff)

    return {k: sp.Rational(v) for k, v in coeffs.items() if v != 0}


def matrix_to_laurent_coeffs(matrix: np.ndarray, symbols: Sequence[sp.Symbol]) -> List[List[PolyCoeffs]]:
    return [
        [expression_to_laurent_coeffs(matrix[row, col], symbols) for col in range(matrix.shape[1])]
        for row in range(matrix.shape[0])
    ]


def infer_power_bounds_from_coeffs(matrix_coeffs: List[List[PolyCoeffs]], symbols: Sequence[sp.Symbol]) -> List[Tuple[int, int]]:
    """Return [(min_power, max_power), ...] for each symbolic variable."""
    if not symbols:
        return []

    mins = [0 for _ in symbols]
    maxs = [0 for _ in symbols]
    saw_any = False

    for row in matrix_coeffs:
        for poly in row:
            for key in poly:
                saw_any = True
                for i, p in enumerate(key):
                    mins[i] = min(mins[i], p)
                    maxs[i] = max(maxs[i], p)

    if not saw_any:
        return [(0, 0) for _ in symbols]
    return list(zip(mins, maxs))


def phi_monomial_keys_from_matrix_bounds(
    bounds: Sequence[Tuple[int, int]],
    input_size: int,
) -> List[MonomialKey]:
    """Return allowed phi monomial powers using the dimension-dependent bound.

    For each symbolic variable, let A use powers in [alpha, beta], where
    alpha and beta already include 0 because infer_power_bounds_from_coeffs()
    initializes the min/max powers at 0.  Let d = beta - alpha and let
    s = input_size = number of columns of A = length of one phi column.

    The search window is

        [-beta - (s - 1)d,  -alpha + (s - 1)d].

    This replaces the old one-step bound [alpha - beta, beta - alpha].
    """
    if not bounds:
        return [tuple()]

    multiplier = max(0, input_size - 1)
    ranges = []
    for alpha, beta in bounds:
        d = beta - alpha
        lo = -beta - multiplier * d
        hi = -alpha + multiplier * d
        ranges.append(range(lo, hi + 1))

    return [tuple(x) for x in product(*ranges)]


def add_keys(a: MonomialKey, b: MonomialKey) -> MonomialKey:
    return tuple(x + y for x, y in zip(a, b))


def z3_sum(items: Iterable[z3.ArithRef]) -> z3.ArithRef:
    items = list(items)
    return sum(items, z3.RealVal(0))


def z3_and(items: Iterable[z3.BoolRef]) -> z3.BoolRef:
    items = list(items)
    return z3.And(items) if items else z3.BoolVal(True)


def z3_or(items: Iterable[z3.BoolRef]) -> z3.BoolRef:
    items = list(items)
    return z3.Or(items) if items else z3.BoolVal(False)


def z3_poly_mul(a: Z3PolyCoeffs, b: Z3PolyCoeffs) -> Z3PolyCoeffs:
    """Multiply two Z3 Laurent polynomials."""
    out: Z3PolyCoeffs = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            key = add_keys(ka, kb)
            out[key] = out.get(key, z3.RealVal(0)) + ca * cb
    return {k: z3.simplify(v) for k, v in out.items()}


def z3_poly_mul_fraction(a: Z3PolyCoeffs, b: PolyCoeffs) -> Z3PolyCoeffs:
    """Multiply a Z3 Laurent polynomial by a rational/SymPy Laurent polynomial."""
    out: Z3PolyCoeffs = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            key = add_keys(ka, kb)
            out[key] = out.get(key, z3.RealVal(0)) + ca * z3_real(cb)
    return {k: z3.simplify(v) for k, v in out.items()}


def z3_poly_vector_dot_fraction(left: Sequence[Z3PolyCoeffs], right: Sequence[PolyCoeffs]) -> Z3PolyCoeffs:
    """Return sum_i left_i * previous_i where previous_i has known coefficients."""
    out: Z3PolyCoeffs = {}
    for a, b in zip(left, right):
        prod = z3_poly_mul_fraction(a, b)
        for key, val in prod.items():
            out[key] = out.get(key, z3.RealVal(0)) + val
    return {k: z3.simplify(v) for k, v in out.items()}


def poly_is_zero_constraints(poly: Z3PolyCoeffs) -> List[z3.BoolRef]:
    return [coeff == 0 for coeff in poly.values()]


def poly_is_nonzero_constraint(poly: Z3PolyCoeffs) -> z3.BoolRef:
    return z3_or(coeff != 0 for coeff in poly.values())


def poly_coeffs_to_sympy(poly: PolyCoeffs, symbols: Sequence[sp.Symbol]) -> sp.Expr:
    expr = sp.Integer(0)
    for key, coeff in poly.items():
        term = sp.Rational(coeff)
        for sym, power in zip(symbols, key):
            term *= sym ** power
        expr += term
    return sp.expand(expr)


def poly_vector_to_sympy_column(vec: List[PolyCoeffs], symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """Convert one accepted phi column from coefficient dictionaries to a SymPy column."""
    return sp.Matrix([poly_coeffs_to_sympy(poly, symbols) for poly in vec])


def previous_vectors_to_matrix(previous: List[List[PolyCoeffs]], symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """Return the matrix P whose columns are the already accepted phi columns."""
    if not previous:
        return sp.zeros(0, 0)
    cols = [poly_vector_to_sympy_column(vec, symbols) for vec in previous]
    return sp.Matrix.hstack(*cols)


def clear_denominators_vector(vec: sp.Matrix, symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """Clear rational-function denominators from a SymPy vector.

    SymPy nullspace is computed over the fraction field F(x), so its basis
    vectors may contain entries like 1/(1 + epsilon).  Multiplying by a common
    denominator produces a Laurent/polynomial vector spanning the same one-
    dimensional witness direction, which is enough for the nonzero dot test.
    """
    denoms = []
    entries = []
    for entry in vec:
        entry = sp.together(sp.sympify(entry))
        num, den = sp.fraction(entry)
        entries.append((num, den))
        denoms.append(den)

    common_den = sp.Integer(1)
    for den in denoms:
        common_den = sp.lcm(common_den, den)

    cleared = []
    for num, den in entries:
        cleared.append(sp.expand(common_den * num / den))
    return sp.Matrix(cleared)


def left_nullspace_witnesses(
    previous: List[List[PolyCoeffs]],
    input_size: int,
    symbols: Sequence[sp.Symbol],
) -> List[List[PolyCoeffs]]:
    """Compute known witnesses q for q^T s = 0 for all accepted columns s.

    This eliminates the symbolic Z3 y/w variables.  Let P be the matrix whose
    columns are the accepted vectors.  A new vector v is independent from P over
    F(x) iff there exists q in null(P.T) with q.T*v != 0.  We compute a basis
    q_1,...,q_r of null(P.T) using SymPy, clear denominators, and later encode
    OR_i(q_i.T * v != 0) in Z3.
    """
    const_key = tuple(0 for _ in symbols)

    if not previous:
        # If no vectors have been selected, v only needs to be nonzero.
        # Standard coordinate witnesses encode OR_i v_i != 0.
        witnesses: List[List[PolyCoeffs]] = []
        for i in range(input_size):
            vec: List[PolyCoeffs] = []
            for j in range(input_size):
                vec.append({const_key: sp.Rational(1)} if i == j else {})
            witnesses.append(vec)
        return witnesses

    P = previous_vectors_to_matrix(previous, symbols)  # input_size x len(previous)
    null_basis = P.T.nullspace()  # q such that P.T*q = 0, over F(x)

    witnesses: List[List[PolyCoeffs]] = []
    for q in null_basis:
        q_clear = clear_denominators_vector(q, symbols)
        witness = [expression_to_laurent_coeffs(entry, symbols) for entry in q_clear]
        # Ignore accidental zero vectors, though SymPy should not return them.
        if any(poly for poly in witness):
            witnesses.append(witness)

    return witnesses


# -----------------------------
# Sparsifier
# -----------------------------

class SymbolicSparsifier:
    def __init__(
        self,
        matrix: np.ndarray,
        name: str,
        symbolic_variable_score: int = SYMBOLIC_VARIABLE_SCORE,
        symbolic_score_mode: str = SYMBOLIC_SCORE_MODE,
    ):
        self.matrix = np.array(matrix, dtype=object)
        self.name = name
        self.solver = z3.Solver()

        # A is t x input_dim, e.g. 20 x 9.  Phi is input_dim x input_dim.
        self.input_size = self.matrix.shape[1]
        self.output_size = self.matrix.shape[0]
        self.symbols = infer_symbolic_variables(self.matrix)
        self.matrix_coeffs = matrix_to_laurent_coeffs(self.matrix, self.symbols)

        self.matrix_power_bounds = infer_power_bounds_from_coeffs(self.matrix_coeffs, self.symbols)
        self.phi_monomial_keys = phi_monomial_keys_from_matrix_bounds(self.matrix_power_bounds, self.input_size)

        self.symbolic_variable_score = int(symbolic_variable_score)
        if symbolic_score_mode not in {"variable", "term"}:
            raise ValueError("symbolic_score_mode must be 'variable' or 'term'")
        self.symbolic_score_mode = symbolic_score_mode

        # solution[col] is one phi entry as a Laurent polynomial in epsilon.
        self.solution: List[Z3PolyCoeffs] = []
        self.res_polys: List[Z3PolyCoeffs] = []
        self.score = 0
        # No symbolic y/w variables are created in this version.
        # Independence is enforced using known SymPy nullspace witnesses.

    def sol_vect(self):
        """Create Laurent-polynomial Z3 variables for one phi column.

        This version intentionally does NOT create symbolic y/w certificate
        variables.  Independence is added separately by add_independence_constraint()
        using known SymPy nullspace witnesses.
        """
        self.solution = []
        for col in range(self.input_size):
            poly: Z3PolyCoeffs = {}
            for key in self.phi_monomial_keys:
                suffix = "_".join(str(p).replace("-", "m") for p in key) if key else "const"
                poly[key] = z3.Real(f"sol_{col}_{suffix}")
            self.solution.append(poly)

    def result_polynomials(self):
        """Compute res[row] = A[row, :] @ sol as Laurent polynomials.

        If A[row,col] has power a and sol[col] has power b, the result has
        power a+b. This is the key change that lets phi itself contain powers.
        """
        self.res_polys = []
        for row in range(self.output_size):
            row_poly: Z3PolyCoeffs = {}
            for col in range(self.input_size):
                A_poly = self.matrix_coeffs[row][col]
                sol_poly = self.solution[col]
                for A_key, A_coeff in A_poly.items():
                    if A_coeff == 0:
                        continue
                    for sol_key, sol_coeff in sol_poly.items():
                        out_key = add_keys(A_key, sol_key)
                        row_poly[out_key] = row_poly.get(out_key, z3.RealVal(0)) + z3_real(A_coeff) * sol_coeff
            self.res_polys.append({k: z3.simplify(v) for k, v in row_poly.items()})

    def increment(self):
        self.score += 1

    def entry_score(self, poly: Z3PolyCoeffs) -> z3.ArithRef:
        coeffs = list(poly.values())
        is_zero = z3_and(c == 0 for c in coeffs)
        const_key = tuple(0 for _ in self.symbols)
        constant_coeff = poly.get(const_key, z3.RealVal(0))
        nonconstant_keys = [key for key in poly if key != const_key]
        nonconstant_coeffs = [poly[key] for key in nonconstant_keys]

        has_only_constant = z3_and(c == 0 for c in nonconstant_coeffs)
        is_pm_one = z3.And(has_only_constant, z3.Or(constant_coeff == 1, constant_coeff == -1))

        base_arithmetic_score = z3.If(is_zero, 0, 1) + z3.If(z3.Or(is_zero, is_pm_one), 0, 1)

        if self.symbolic_score_mode == "term":
            # In term mode, each nonzero nonconstant Laurent monomial
            # contributes one SYMBOLIC_VARIABLE_SCORE.
            # Example: epsilon**-1 + epsilon**4 contributes 2 * SYMBOLIC_VARIABLE_SCORE.
            symbolic_score = z3_sum(
                z3.If(poly[key] != 0, self.symbolic_variable_score, 0)
                for key in nonconstant_keys
            )
        else:
            symbolic_score_terms = []
            for sym_idx, _sym in enumerate(self.symbols):
                uses_symbol = z3_or(
                    poly[key] != 0
                    for key in nonconstant_keys
                    if key[sym_idx] != 0
                )
                symbolic_score_terms.append(z3.If(uses_symbol, self.symbolic_variable_score, 0))
            symbolic_score = z3_sum(symbolic_score_terms)

        return base_arithmetic_score + symbolic_score

    def sparsify(self):
        total_score = z3_sum(self.entry_score(poly) for poly in self.res_polys)
        self.solver.add(total_score == self.score)

    def checkpoint(self):
        self.solver.push()

    def erase(self):
        self.solver.pop()

    def add_independence_constraint(self, previous: List[List[PolyCoeffs]]):
        """Constrain the current symbolic phi column to be independent of previous.

        Let P be the matrix of previously accepted columns.  Instead of searching
        for an unknown certificate y/w in Z3, compute known witnesses q spanning
        null(P.T) over F(x), then require OR_q q.T * sol != 0.
        This is equivalent to sol not being in span_{F(x)}(previous).
        """
        witnesses = left_nullspace_witnesses(previous, self.input_size, self.symbols)

        if not witnesses:
            # This means previous already spans the whole input space, so no
            # further independent vector can exist.
            self.solver.add(z3.BoolVal(False))
            return

        nonzero_tests = []
        for q in witnesses:
            dot_poly = z3_poly_vector_dot_fraction(self.solution, q)
            nonzero_tests.append(poly_is_nonzero_constraint(dot_poly))

        self.solver.add(z3_or(nonzero_tests))

    def model_solution_as_poly_vector(self, model: z3.ModelRef) -> List[PolyCoeffs]:
        out: List[PolyCoeffs] = []
        for poly in self.solution:
            coeffs: PolyCoeffs = {}
            for key, var in poly.items():
                val = z3_value_to_fraction(model.eval(var, model_completion=True))
                if val != 0:
                    coeffs[key] = sp.Rational(val.numerator, val.denominator)
            out.append(coeffs)
        return out

    def base_run(self, previous: List[List[PolyCoeffs]]):
        self.sol_vect()
        self.result_polynomials()
        self.add_independence_constraint(previous)
        self.increment()
        self.checkpoint()
        self.sparsify()


def find_vectors(matrix: np.ndarray) -> List[List[PolyCoeffs]]:
    s = SymbolicSparsifier(matrix, array_name)
    solutions: List[List[PolyCoeffs]] = []
    s.base_run(solutions)

    print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}  "
          f"(t x nm — looking for {matrix.shape[1]}x{matrix.shape[1]} phi)")
    print(f"symbolic variables: {[str(x) for x in s.symbols]}")
    print(f"matrix power bounds: {s.matrix_power_bounds}")
    print(f"Laurent bound multiplier s-1: {s.input_size - 1}")
    print(f"phi monomial keys: {s.phi_monomial_keys}")
    print(f"phi variables per entry: {len(s.phi_monomial_keys)}")
    print("independence mode: SymPy nullspace witnesses, no Z3 y/w variables")
    print(f"total Z3 sol variables per phi column: {s.input_size * len(s.phi_monomial_keys)}")
    print(f"symbolic score per variable/term: {s.symbolic_variable_score}")
    print(f"symbolic score mode: {s.symbolic_score_mode}")

    while len(solutions) != matrix.shape[1]:
        if s.solver.check() == z3.sat:
            model = s.solver.model()
            sol = s.model_solution_as_poly_vector(model)
            solutions.append(sol)
            sol_pretty = [poly_coeffs_to_sympy(p, s.symbols) for p in sol]
            print(f"score {s.score}: {sol_pretty}")
            s.erase()
            s.add_independence_constraint(solutions)
            s.checkpoint()
            s.sparsify()
        else:
            s.erase()
            s.increment()
            s.checkpoint()
            s.sparsify()
            print(f"trying score {s.score}")

    return solutions


# -----------------------------
# Evaluation / saving helpers
# -----------------------------

def to_sympy_matrix(matrix: np.ndarray) -> sp.Matrix:
    return sp.Matrix([[sp.sympify(matrix[i, j]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])])


def poly_vectors_to_phi_matrix(poly_vectors: List[List[PolyCoeffs]], symbols: Sequence[sp.Symbol]) -> sp.Matrix:
    """poly_vectors are phi columns; return phi as input_dim x input_dim."""
    cols = []
    for vec in poly_vectors:
        cols.append(sp.Matrix([poly_coeffs_to_sympy(poly, symbols) for poly in vec]))
    return sp.Matrix.hstack(*cols)


def expression_score(expr, symbols: Sequence[sp.Symbol]) -> int | float:
    coeffs = expression_to_laurent_coeffs(expr, symbols)
    if not coeffs:
        return 0

    const_key = tuple(0 for _ in symbols)
    nonconstant_keys = [key for key in coeffs if key != const_key]
    only_constant = len(nonconstant_keys) == 0
    is_pm_one = only_constant and coeffs.get(const_key, 0) in {sp.Integer(1), sp.Integer(-1)}

    score = 1 + (0 if is_pm_one else 1)
    if SYMBOLIC_SCORE_MODE == "term":
        # Each distinct nonconstant Laurent monomial contributes one symbolic penalty.
        score += len(nonconstant_keys) * SYMBOLIC_VARIABLE_SCORE
    else:
        for sym_idx, _sym in enumerate(symbols):
            if any(key[sym_idx] != 0 for key in nonconstant_keys):
                score += SYMBOLIC_VARIABLE_SCORE
    return score


def count_arithmetic_or_symbolic_score(matrix: sp.Matrix, symbols: Sequence[sp.Symbol]) -> int | float:
    base_offset = -matrix.shape[1] if array_name == "w" else -matrix.shape[0]
    return base_offset + sum(
        expression_score(matrix[row, col], symbols)
        for row in range(matrix.rows)
        for col in range(matrix.cols)
    )


def save_sympy_matrix_csv(path: Path, matrix: sp.Matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(matrix.rows):
            f.write(",".join(str(sp.expand(matrix[i, j])) for j in range(matrix.cols)) + "\n")


if __name__ == "__main__":
    array_chosen = load_input_matrix()
    symbols = infer_symbolic_variables(array_chosen)

    print(array_chosen)
    print(name)
    print("=======================")

    phi_columns = find_vectors(array_chosen)
    phi_sp = poly_vectors_to_phi_matrix(phi_columns, symbols)

    print("phi =")
    print(phi_sp)
    print(f"det(phi): {sp.factor(phi_sp.det())}")

    transformed = to_sympy_matrix(array_chosen) * phi_sp
    transformed = transformed.applyfunc(sp.expand)
    print("transformed matrix =")
    print(transformed)

    original_score = count_arithmetic_or_symbolic_score(to_sympy_matrix(array_chosen), symbols)
    new_score = count_arithmetic_or_symbolic_score(transformed, symbols)
    print(f"original score: {original_score}")
    print(f"resulting score: {new_score}")

    if phi_sp.det() != 0:
        phi_inv = phi_sp.inv().applyfunc(sp.expand)
        reconstruction_ok = to_sympy_matrix(array_chosen).equals(transformed * phi_inv)
        print(f"reconstruction ok: {reconstruction_ok}")
    else:
        phi_inv = None
        print("phi is singular over SymPy; no inverse saved/computed.")

    if SAVE_OUTPUTS:
        out_dir = BASE_DIR / algorithmFolder / name / "Optimal"
        if phi_inv is not None:
            save_sympy_matrix_csv(out_dir / f"{name}_{array_name}_arr_phi_inv.csv", phi_inv)
        save_sympy_matrix_csv(out_dir / f"new_{name}_{array_name}_arr.csv", transformed)
