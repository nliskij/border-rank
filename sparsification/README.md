# Symbolic Sparsification Algorithm
This folder contains the sparsification results and the algorithm used for sparsification. The "all_results" folder contains all algorithms that we have sparsified, and "best_results" are the selected best for a specific algorithm dimension size, with best in regards to leading coefficient. 

The python file "sparsification_alg.py" contains the script used for sparsification, and "smirnov_333_20_encoding_literals.py" is a sample of what the input script for the "MATRIX_FILE" field in the sparsification algorithm is supposed to look like. The sparsification algorithm file also includes support for non-approximate algorithms. 

# Steps to run the sparsification algorithm
1. Specify the "name," "algorithmFolder," and "array_name" fields. Only the latter affects the algorithm, where you choose which matrix (u, v, w) to sparsify. 
2. Specify USE_SMIRNOV_SYMBOLIC_INPUT. This is true if you are sparsifying an APA algorithm, and it is in the format of "smirnov_333_20_encoding_literals.py."
3. Specify MATRIX_FILE, which is the name to the file of the algorithm that you are trying to sparsify. 
4. Specify SYMBOLIC_VARIABLE_SCORE, SYMBOLIC_SCORE_MODE. 
5. Specify if you want to save the outputs with WAVE_OUTPUTS.
6. Specify your base directory with BASE_DIR.
7. Run it with py sparsification_alg.py (or whatever alternative you use to run a script with python)