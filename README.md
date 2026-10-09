# APA Matrix Multiplication Algorithms 

This repository contains (symbolic) encoding matrices for several new arbitrary precision approximate (APA) matrix multiplication algorithms. These we discovered through a novel search algorithm, which we also include code for. For further background, see the paper

I. Dumitiru, O. Holtz, A. Hsu, N. Liskij, Y. Moran, R. Schneider, O. Schwartz, and S. Yuan. Fast and Stable Arbitrary Precision Approximate Matrix Multiplication. To appear.

# How to use this repository

Schemes are grouped by the canonical nondecreasing dimension triple.  Each leaf
directory contains exactly:

- `decomposition.exp`: an exact rank decomposition of `Tr(ABC)`;
- `witness.json`: the minimal term/coefficient data and one one-parameter
  subgroup.

The witness does not repeat dimensions, rank, the resulting bound, or
literature metadata.  Those values are inferred from the `.exp` file and the
witnessed reduction.

Run the complete check with:

```text
python3 verify_border_rank.py border_rank_cases
```

To verify one pair directly:

```text
python3 verify_border_rank.py decomposition.exp witness.json
```

The verifier first proves that the `.exp` file is an exact decomposition of
`Tr(ABC)`.  If `E_A,E_B,E_C` are the stored basis matrices and `D_A(t),
D_B(t),D_C(t)` are the diagonal matrices defined by the stored weights, it
then applies

```text
U -> D_A E_A^-1 U E_B D_B^-1
V -> D_B E_B^-1 V E_C D_C^-1
W -> D_C E_C^-1 W E_A D_A^-1
```

to every listed term and verifies positive total `t`-order.  For a
`nullcone_combination`, it first expands the weighted sum of rank-one tensors
exactly (including cancellations) and checks that the whole sum has positive
order.  Such a witness reduces the border-rank bound by one.  For
`deleted_terms`, the same bases and weights must contract every listed term,
so this checks simultaneous vanishing rather than separate nilpotence.

## Stored bounds

| size | exact rank | reduction | resulting bound | literature border bound | gap |
|---|---:|---:|---:|---:|---:|
| (2,2,2) | 8 | 1 | 7 | 7 | 0 |
| (2,2,3) | 11 | 1 | 10 | 10 | 0 |
| (2,2,5) | 18 | 1 | 17 | 16 | 1 |
| (2,2,6) | 22 | 2 | 20 | 19 | 1 |
| (2,2,8) | 29 | 2 | 27 | 26 | 1 |
| (2,3,3) | 16 | 1 | 15 | 14 | 1 |
| (2,3,5) | 26 | 2 | 24 | 23 | 1 |
| (2,3,6) | 30 | 1 | 29 | 28 | 1 |
| (2,3,7) | 35 | 1 | 34 | 33 | 1 |
| (2,4,4) | 26 | 1 | 25 | 24 | 1 |
| (2,4,5) | 33 | 2 | 31 | 32 | -1 |
| (2,4,6) | 39 | 2 | 37 | 37 | 0|
| (2,5,5) | 40 | 1 | 39 | 39 | 0 |
| (3,3,3) | 23 | 2 | 21 | 20 | 1 |
| (3,3,4) | 29 | 2 | 27 | 27 | 0 |
| (3,3,5) | 36 | 1 | 35 | 34 | 1 |
| (3,3,7) | 49 | 1 | 48 | 47 | 1 |
| (3,3,8) | 55 | 1 | 54 | 53 | 1 |
| (3,4,5) | 48 | 2 | 46 | 45 | 1 |
| (3,5,6) | 68 | 1 | 67 | 67 | 0 |
| (3,6,8) | 108 | 1 | 107 | 106 | 1 |

Bounds from the literature are taken from: A. V. Smirnov. The Bilinear Complexity and Practical Algorithms for Matrix Multiplication. _Computational Mathematics and Mathematical Physics_ (2013).
