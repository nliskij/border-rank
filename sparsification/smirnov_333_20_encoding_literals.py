import sympy as sp

epsilon = sp.symbols("epsilon")

U = [
    sp.Matrix([
        [0, -epsilon**2, epsilon**4],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [-epsilon, -epsilon, 0],
        [0, 1, 0],
        [epsilon**(-2), epsilon, -1],
    ]),
    sp.Matrix([
        [epsilon, -epsilon**4 + epsilon, 0],
        [0, 0, 0],
        [-1/epsilon**2, 0, 1],
    ]),
    sp.Matrix([
        [-1/epsilon, 0, epsilon**4],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, epsilon**2, 0],
        [-1/epsilon**2, 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [1, 0, -epsilon**3 - epsilon**2],
        [-1, 0, epsilon**2],
        [-1/epsilon**2, 0, 1],
    ]),
    sp.Matrix([
        [-1/epsilon, 0, epsilon],
        [0, 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, epsilon**2 - epsilon, 0],
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
    ]),
    sp.Matrix([
        [-epsilon, -epsilon, 0],
        [0, 1 - epsilon**3, 0],
        [epsilon**(-2), 0, -1],
    ]),
    sp.Matrix([
        [epsilon, epsilon, 0],
        [-1, 0, epsilon**2],
        [-1/epsilon**2, 0, 1],
    ]),
    sp.Matrix([
        [0, -epsilon, 0],
        [0, 0, 0],
        [epsilon**(-2), epsilon, 0],
    ]),
    sp.Matrix([
        [0, epsilon, 0],
        [0, 0, 0],
        [-1/epsilon**2, 0, 0],
    ]),
    sp.Matrix([
        [-epsilon, epsilon**4, 0],
        [0, 0, 0],
        [epsilon**(-2), 0, -1],
    ]),
    sp.Matrix([
        [1/epsilon, -epsilon, 0],
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
    ]),
    sp.Matrix([
        [0, -epsilon**3, 0],
        [1/epsilon, -1, epsilon**4],
        [-1/epsilon**2, -epsilon, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, epsilon],
        [epsilon**(-2), 0, -1],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, epsilon**2, 0],
        [0, 1, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 1, 0],
        [epsilon**(-2), epsilon, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [epsilon**(-2), epsilon, 0],
        [0, 0, 0],
    ]),
]

V = [
    sp.Matrix([
        [epsilon, 0, -epsilon**3],
        [0, 0, 0],
        [1/epsilon, 0, epsilon**(-2)],
    ]),
    sp.Matrix([
        [0, 0, epsilon**3],
        [epsilon**(-2), -1/epsilon, 0],
        [0, 0, -epsilon],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [-1/epsilon, 1, 0],
        [0, -1/epsilon**2, 0],
    ]),
    sp.Matrix([
        [-1, epsilon, 0],
        [0, 0, 0],
        [-1/epsilon**2, 1/epsilon, 0],
    ]),
    sp.Matrix([
        [0, 0, -epsilon**3],
        [epsilon**(-2), 0, 0],
        [0, 0, epsilon**(-2)],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [epsilon**(-2), -1/epsilon, 0],
    ]),
    sp.Matrix([
        [epsilon, -2*epsilon**2, 0],
        [0, 0, 0],
        [epsilon**(-2), -1/epsilon, 0],
    ]),
    sp.Matrix([
        [epsilon, 0, -epsilon**3],
        [-1/epsilon**2, 0, 1],
        [1/epsilon, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [-1/epsilon**2, 1/epsilon, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, epsilon**(-2), 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [epsilon**(-2), 0, -1],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, epsilon**2, 0],
        [-1/epsilon**2, 0, 1],
        [0, 1, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [-1, epsilon, 0],
        [epsilon**(-2), -1/epsilon, 0],
    ]),
    sp.Matrix([
        [-1, epsilon, epsilon**2],
        [0, 0, 0],
        [-1/epsilon**2, 1/epsilon, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, epsilon**(-2)],
    ]),
    sp.Matrix([
        [0, -epsilon**2, 0],
        [epsilon**(-2), 0, 0],
        [0, -1, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [1/epsilon, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [-1/epsilon, 1, epsilon],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [-1, 0, epsilon**2],
        [0, 0, epsilon**(-2)],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
]

W = [
    sp.Matrix([
        [1, epsilon, 0],
        [1/epsilon, 0, 0],
        [epsilon**(-2), 0, 2*epsilon**3],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, epsilon**3],
        [0, 0, 1/epsilon],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [1/epsilon, 1, -epsilon**2],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [epsilon, 0, 0],
        [1, 0, 0],
        [1/epsilon, 0, 0],
    ]),
    sp.Matrix([
        [1, epsilon, 0],
        [1/epsilon, 0, 0],
        [epsilon**(-2), 1/epsilon, epsilon**3],
    ]),
    sp.Matrix([
        [0, 0, epsilon**2],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [epsilon, 0, epsilon**3],
        [1/epsilon, 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, epsilon],
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, epsilon, 0],
        [0, 0, 1/epsilon],
    ]),
    sp.Matrix([
        [0, 0, epsilon**3],
        [1/epsilon, 1, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, epsilon],
        [0, 0, 1],
        [0, 1, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, -1],
        [epsilon**(-2), 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [epsilon**(-2), 0, 0],
        [0, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [1/epsilon, 0, 0],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, epsilon**(-2), -epsilon**2],
    ]),
    sp.Matrix([
        [1, 0, 0],
        [1/epsilon, -1, 0],
        [epsilon**(-2), 0, 0],
    ]),
    sp.Matrix([
        [0, -epsilon, epsilon**3],
        [1/epsilon, 0, 0],
        [0, 0, -2*epsilon**3],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1/epsilon],
    ]),
    sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, epsilon**(-2), 0],
    ]),
    sp.Matrix([
        [0, epsilon, 0],
        [0, 1, 0],
        [0, 1/epsilon, epsilon**3],
    ]),
]

a = {(i, j): sp.symbols(f"a{i}{j}") for i in range(1, 4) for j in range(1, 4)}
b = {(i, j): sp.symbols(f"b{i}{j}") for i in range(1, 4) for j in range(1, 4)}
c = {(i, j): sp.symbols(f"c{i}{j}") for i in range(1, 4) for j in range(1, 4)}

A_matrix = sp.Matrix([[a[(i, j)] for j in range(1, 4)] for i in range(1, 4)])
B_matrix = sp.Matrix([[b[(i, j)] for j in range(1, 4)] for i in range(1, 4)])
C_matrix = sp.Matrix([[c[(i, j)] for j in range(1, 4)] for i in range(1, 4)])


def frobenius_dot(X, Y):
    return sp.expand(sum(X[i, j] * Y[i, j] for i in range(3) for j in range(3)))


def build_trilinear_tensor(U_list, V_list, W_list):
    return sp.expand(
        sum(
            frobenius_dot(Ui, A_matrix) * frobenius_dot(Vi, B_matrix) * frobenius_dot(Wi, C_matrix)
            for Ui, Vi, Wi in zip(U_list, V_list, W_list)
        )
    )


def build_matrix_product(U_list, V_list, W_list):
    return sp.Matrix(
        [
            [
                sp.expand(
                    sum(
                        frobenius_dot(Ui, A_matrix) * frobenius_dot(Vi, B_matrix) * Wi.T[r, c]
                        for Ui, Vi, Wi in zip(U_list, V_list, W_list)
                    )
                )
                for c in range(3)
            ]
            for r in range(3)
        ]
    )


def min_epsilon_power(expr):
    expr = sp.expand(expr)
    if expr == 0:
        return sp.oo
    vals = []
    for term in sp.Add.make_args(expr):
        vals.append(term.as_powers_dict().get(epsilon, sp.Integer(0)))
    return min(vals)


def has_negative_powers(expr):
    return min_epsilon_power(expr) < 0


T_from_encodings = build_trilinear_tensor(U, V, W)
T_target = sp.expand(sp.trace(A_matrix * B_matrix * C_matrix))
trilinear_error = sp.expand(T_from_encodings - T_target)

matrix_product_from_encodings = build_matrix_product(U, V, W)
matrix_product_error = sp.Matrix(
    [
        [
            sp.expand(matrix_product_from_encodings[r, c] - (A_matrix * B_matrix)[r, c])
            for c in range(3)
        ]
        for r in range(3)
    ]
)


def verify_constant_term():
    return sp.simplify(T_from_encodings.coeff(epsilon, 0) - T_target) == 0


def verify_trilinear_polynomial_error():
    return not has_negative_powers(trilinear_error)


def verify_matrix_polynomial_error():
    return all(not has_negative_powers(entry) for entry in matrix_product_error)


if __name__ == "__main__":
    print("Number of rank-one terms:", len(U))
    print("Constant term is Tr(ABC):", verify_constant_term())
    print("Trilinear error is polynomial in epsilon:", verify_trilinear_polynomial_error())
    print("Matrix-product error is polynomial in epsilon:", verify_matrix_polynomial_error())
