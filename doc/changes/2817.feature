Introduces operator overloading for functions `mean_nonzero` and `mean_abs_nonzero` (mean of absolute values) for complex matrices (qutip.data.Dia, qutip.data.CSR, qutip.data.Dense).
The main difference to a classical mean value function is that `mean_nonzero` and `mean_abs_nonzero` take into account _only non-zero entries_ of input matrices.
