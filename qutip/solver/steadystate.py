from qutip import vector_to_operator, operator_to_vector, hilbert_dist
from qutip import settings, CoreOptions
import qutip.core.data as _data
import numpy as np
import scipy.sparse.csgraph
import scipy.sparse.linalg
from warnings import warn


__all__ = ["steadystate", "steadystate_floquet", "pseudo_inverse"]


def _permute_wbm(L, b):
    perm = np.argsort(
        scipy.sparse.csgraph.maximum_bipartite_matching(L.as_scipy())
    )
    L = _data.permute.indices(L, perm, None, dtype=type(L))
    b = _data.permute.indices(b, perm, None, dtype=type(b))
    return L, b


def _permute_rcm(L, b):
    perm = np.argsort(scipy.sparse.csgraph.reverse_cuthill_mckee(L.as_scipy()))
    L = _data.permute.indices(L, perm, perm, dtype=type(L))
    b = _data.permute.indices(b, perm, None, dtype=type(b))
    return L, b, perm


def _reverse_rcm(rho, perm):
    rev_perm = np.argsort(perm)
    rho = _data.permute.indices(rho, rev_perm, None, dtype=type(rho))
    return rho

# TODO: think of a more maintainable way to contain all information about available methods & solvers. Apparently there is also an implicit mapping of solver names by user and the ones corresponding to actual functions
# TODO: possible kwargs must be much better documented. Drawbacks: might be difficult to maintain since we rely on numpy and scipy functions' interfaces. Is there an automatized approach to it? 

def steadystate(A, c_ops=[], *, method='direct', solver=None, **kwargs):
    """
    Calculates the steady state for quantum evolution subject to the supplied
    Hamiltonian or Liouvillian operator and (if given a Hamiltonian) a list of
    collapse operators.

    If the user passes a Hamiltonian then it, along with the list of collapse
    operators, will be converted into a Liouvillian operator in Lindblad form.

    Parameters
    ----------
    A : :obj:`.Qobj`
        A Hamiltonian or Liouvillian operator.

    c_op_list : list
        A list of collapse operators.

    method : str, {"direct", "eigen", "svd", "power"}, default: "direct"
        The allowed methods are composed of 2 parts, the steadystate method:

        - "direct": Solving ``L(rho_ss) = 0``
        - "eigen" : Eigenvalue problem
        - "svd" : Singular value decomposition
        - "power" : Inverse-power method
        - "propagator" : Repeatedly applying the propagator

    solver : str, optional
        'direct' and 'power' methods only.
        Solver to use when solving the ``L(rho_ss) = 0`` equation.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve"
          sparse solver by mkl.

        Extension to qutip, such as qutip-tensorflow, can use come with their
        own solver. When ``A`` and ``c_ops`` use these data backends, see the
        corresponding libraries ``linalg`` for available solver.

        Extra options for these solver can be passed in ``**kw``.

    use_rcm : bool, default: False
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the LU
        factorization of the Liouvillian.
        Used with 'direct' or 'power' method.

    use_wbm : bool, default: False
        Use Weighted Bipartite Matching reordering to make the Liouvillian
        diagonally dominant.  This is useful for iterative preconditioners
        only. Used with 'direct' or 'power' method.

    weight : float, optional
        Sets the size of the elements used for adding the unity trace condition
        to the linear solvers.  This is set to the average abs value of the
        Liouvillian elements if not specified by the user.
        Used with 'direct' method.

    power_tol : float, default: 1e-12
        Tolerance for the solution when using the 'power' method.

    power_maxiter : int, default: 10
        Maximum number of iteration to use when looking for a solution when
        using the 'power' method.

    power_eps: double, default: 1e-15
        Small weight used in the "power" method.

    sparse: bool, default: True
        Whether to use the sparse eigen solver with the "eigen" method
        (default sparse).  With "direct" and "power" method, when the solver is
        not specified, it is used to set whether "solve" or "spsolve" is
        used as default solver.

    rho: Qobj, default: None
        Initial state for the "propagator" method.

    propagator_T: float, default: 10
        Initial time step for the propagator method. The time step is doubled
        each iteration.

    propagator_tol: float, default: 1e-5
        Tolerance for propagator method convergence. If the Hilbert distance
        between the states of a step is less than this tolerance, the state is
        considered to have converged to the steady state.

    propagator_max_iter: int, default: 30
        Maximum number of iterations until convergence. A RuntimeError is
        raised if the state did not converge.

    **kwargs :
        Extra options to pass to the linear system solver. See the
        documentation of the used solver in ``numpy.linalg`` or
        ``scipy.sparse.linalg`` to see what extra arguments are supported.

    Returns
    -------
    dm : qobj
        Steady state density matrix.
    info : dict, optional
        Dictionary containing solver-specific information about the solution.

    Notes
    -----
    The SVD method works only for dense operators (i.e. small systems).
    """
    if not A.issuper and not c_ops:
        raise TypeError('Cannot calculate the steady state for a ' +
                        'non-dissipative system.')
    if not A.issuper:
        A = liouvillian(A, c_ops)
    else:
        for op in c_ops:
            A += lindblad_dissipator(op)

    if "-" in method:
        # to support v4's "power-gmres" method
        method, solver = method.split("-")

    if solver == "mkl":
        solver = "mkl_spsolve"

    # Keys supported in v4, but removed in v5
    if kwargs.pop("return_info", False):
        warn("Steadystate no longer supports return_info", DeprecationWarning)
    if "mtol" in kwargs and "power_tol" not in kwargs: # TODO: Lacks explanation for this argument
        kwargs["power_tol"] = kwargs["mtol"]
    kwargs.pop("mtol", None) # TODO: can be moved to the previous if statement, unnecessary line

    if method == "eigen": # TODO: would it be neater if we use "case" syntax
        return _steadystate_eigen(A, **kwargs)
    if method == "svd":
        return _steadystate_svd(A, **kwargs)

    # TODO: I think the entire logic here can be refactored into a separate entity performing 1) checks on data types + conversions, 2) matching methods and kwargs
    # We want to be able to use this without having to know what data type the
    # liouvillian uses. For extra data types (tensorflow) we can expect
    # the users to know they are using them and choose an appropriate solver
    sparse_solvers = ["spsolve", "mkl_spsolve", "gmres", "lgmres", "bicgstab"] # TODO: Can be refactored into an enumerator or so. Can also become an attribute of the class to be ablr to generate documentation about supported methods more automatically and impose validation in case of non supported methods
    if not isinstance(A.data, (_data.CSR, _data.Dense)): # TODO: pass doesn't give any value to this check. Either give a warning to the user, or do something here
        # Tensorflow, jax, etc. data type
        pass
    elif isinstance(A.data, _data.CSR) and solver in ["solve", "lstsq"]: # Here we start to handle conversions (a lot of repeatable logic) into different data types based on a solver chosen by a user.
        A = A.to("dense")
    elif isinstance(A.data, _data.Dense) and solver in sparse_solvers:
        A = A.to("csr")
    elif solver is None and kwargs.get("sparse", False):
        A = A.to("csr")
        solver = "mkl_spsolve" if settings.has_mkl else "spsolve"
    elif solver is None and (kwargs.get("sparse", None) is False):
        # sparse is explicitly set to false, v4 tag to use `numpy.linalg.solve`
        A = A.to("dense")
        solver = "solve"

    if method in ["direct", "iterative"]:
        # Remove unused kwargs, so only used and pass-through ones are included
        kwargs.pop("power_tol", 0) # TODO: why to even remove them?
        kwargs.pop("power_maxiter", 0)
        kwargs.pop("power_eps", 0)
        kwargs.pop("sparse", 0)
        with CoreOptions(default_dtype_scope="creation"):
            # We want to ensure the dtype we set are kept
            rho_ss = _steadystate_direct(A, kwargs.pop("weight", 0),
                                         method=solver, **kwargs)

    elif method == "power":
        # Remove unused kwargs, so only used and pass-through ones are included
        kwargs.pop("weight", 0)
        kwargs.pop("sparse", 0)
        with CoreOptions(default_dtype_scope="creation"):
            # We want to ensure the dtype we set are kept
            rho_ss = _steadystate_power(A, method=solver, **kwargs)

    elif method == "propagator":
        rho_ss = _steadystate_expm(A, **kwargs)
    else:
        raise ValueError(f"method {method} not supported.") # TODO: validation can be performed automatically if we create an enumerator/model for supported solvers and methods

    return rho_ss


def _steadystate_direct(A: Qobj, weight: float, **kw): # TODO: I'd suggest to expand the keyword arguments here too
    # Convert Dia to CSR for cleaner diagonal matrix representation:
    # without zeros or uninitialised padded elements, which is especially
    # relevant for multi-diagonal cases
    if isinstance(A.data, _data.Dia):
        A = A.to("csr")

    if not weight:
      # Calculate weight if not provided by user
      # (currently, no good dispatched function is available)
      if isinstance(A.data, _data.CSR):
          weight = np.mean(np.abs(A.data.as_scipy().data)) # TODO: will be refactored to use mean_abs_nonzero. No need to check for data type!
      else:
          A_np = np.abs(A.data.to_array()) # TODO: same refactoring
          weight = np.mean(A_np[A_np > 0])

    # Add weight to the Liouvillian
    # L[:, 0] = A[:, 0] + vectorized(eye * weight).T
    # L[:, 1:] = A[:, 1:]
    N = A.shape[0]
    n = int(N**0.5) # TODO: what is this parameter?
    dtype = type(A.data)
    if dtype == _data.Dia: # TODO: from this and similar checks (see above) I conclude the desired format is either CSR or Dense. This line is also redundant since there is Dia -> CSR conversion happening at the beginning.
        # Dia is bad at vector and missing optimization such as `use_wbm`.
        dtype = _data.CSR
    weight_vec = _data.column_stack(_data.diag([weight] * n, 0, dtype=dtype))
    first_row = _data.block_extract(A.data, 0, 1, 0, N, dtype=dtype)
    L = _data.block_overwrite(
        A.data, _data.add(first_row, weight_vec.transpose()), 0, 0, dtype=dtype
    )
    b = _data.one_element[dtype]((N, 1), (0, 0), weight) # TODO: What is the final equation this all gets composed into?

    # Permutation are part of scipy.sparse, thus only supported for CSR.
    if kw.pop("use_wbm", False):
        if isinstance(L, _data.CSR):
            L, b = _permute_wbm(L, b)
        else:
            warn("Only CSR matrices can be permuted.", RuntimeWarning)
    use_rcm = False # TODO: this could be embedded into signature
    if kw.pop("use_rcm", False):
        if isinstance(L, _data.CSR): # TODO: this means we always use permutation in case of CSR
            L, b, perm = _permute_rcm(L, b)
            use_rcm = True
        else:
            warn("Only CSR matrices can be permuted.", RuntimeWarning)
    if kw.pop("use_precond", False):
        if isinstance(L, (_data.CSR, _data.Dia)): # TODO: if we already converted Dia to CSR, the check for Dia is redundant
            kw["M"] = _compute_precond(L, kw) # TODO: this means we always use preconditioners if the matrix is Dia or CSR. Hence, we cannot just convert Dia to CSR at the beginnning (which we do!)
            # TODO: info is lacking on which solver "M" argument corresponds to
        else:
            warn("Only sparse solver use preconditioners.", RuntimeWarning)

    method = kw.pop("method", None)
    steadystate = _data.solve(L, b, method, options=kw)

    if use_rcm: # TODO: will not be used for Dense
        steadystate = _reverse_rcm(steadystate, perm)
    
    # Density matrix
    rho_ss = _data.column_unstack(steadystate, n)
    rho_ss = _data.add(rho_ss, rho_ss.adjoint()) * 0.5 # TODO: for me: adjoint state is conjugate-transpose

    return Qobj(rho_ss, dims=A._dims[0].oper, isherm=True)


def _steadystate_eigen(L, **kw):
    val, vec = (L.dag() @ L).eigenstates(
        eigvals=1,
        sort="low",
        # v4's implementation only uses sparse eigen solver
        sparse=kw.pop("sparse", True)
    )
    rho = vector_to_operator(vec[0])
    return rho / rho.tr()


def _steadystate_svd(L, **kw):
    N = L.shape[0]
    n = int(N**0.5)
    u, s, vh = _data.svd(L.data, True)
    vec = _data.split_columns(vh.adjoint())[-1]
    rho = _data.column_unstack(vec, n)
    rho = Qobj(rho, dims=L._dims[0].oper, isherm=True)
    return rho / rho.tr()


def _steadystate_expm(L, rho=None, propagator_tol=1e-5, propagator_T=10, **kw):
    if rho is None:
        from qutip import rand_dm
        rho = rand_dm(L._dims[0].oper[0])
    # Propagator at an arbitrary long time
    prop = (propagator_T * L).expm()

    niter = 0
    max_iter = kw.get("propagator_max_iter", 30)
    while niter < max_iter:
        rho_next = prop(rho)
        rho_next = (rho_next + rho_next.dag()) / (2 * rho_next.tr())
        if np.real(hilbert_dist(rho_next, rho)) <= propagator_tol:
            return rho_next
        rho = rho_next
        prop = prop @ prop
        niter += 1

    raise RuntimeError(
        f"Did not converge to a steadystate after {max_iter} iterations."
    )


def _steadystate_power(A, **kw): # TODO: type hints, expand kwargs
    A += kw.pop("power_eps", 1e-15) # TODO: why does it get added
    L = A.data
    N = L.shape[1]
    y = _data.Dense([1]*N)

    # Permutation are part of scipy.sparse, thus only supported for CSR. # TODO: repeatable logic, same as in direct method
    if kw.pop("use_wbm", False):
        if isinstance(L, _data.CSR):
            L, y = _permute_wbm(L, y)
        else:
            warn("Only CSR matrices can be permuted.", RuntimeWarning)
    use_rcm = False
    if kw.pop("use_rcm", False):
        if isinstance(L, _data.CSR):
            L, y, perm = _permute_rcm(L, y)
            use_rcm = True
        else:
            warn("Only CSR matrices can be permuted.", RuntimeWarning)
    if kw.pop("use_precond", False):
        if isinstance(L, (_data.CSR, _data.Dia)):
            kw["M"] = _compute_precond(L, kw)
        else:
            warn("Only sparse solver use preconditioners.", RuntimeWarning)

    it = 0
    maxiter = kw.pop("power_maxiter", 10) # TODO: think how to unpack all arguments with their default values more elegantly
    tol = kw.pop("power_tol", 1e-12)
    method = kw.pop("method", None)
    while it < maxiter and _data.norm.max(L @ y) > tol:
        y = _data.solve(L, y, method, options=kw)
        y = y / _data.norm.max(y)
        it += 1

    if it >= maxiter:
        raise Exception('Failed to find steady state after ' +
                        str(maxiter) + ' iterations')

    if use_rcm:
        y = _reverse_rcm(y, perm)

    rho_ss = Qobj(_data.column_unstack(y, N**0.5), dims=A._dims[0].oper)
    rho_ss = rho_ss + rho_ss.dag() # TODO: for me: dag is Hermitian adjoint of the quantum object
    rho_ss = rho_ss / rho_ss.tr()
    rho_ss.isherm = True
    return rho_ss

# TODO: somewhat of a doppelgaenger of steadystate function w.r.t. to arguments. It would also benefit from reusing some common structures (e.g. with information about solvers)
def steadystate_floquet(H_0, c_ops, Op_t, w_d=1.0, n_it=3, sparse=False,
                        solver=None, **kwargs):
    """
    Calculates the effective steady state for a driven
     system with a time-dependent cosinusoidal term:

    .. math::

        \\mathcal{\\hat{H}}(t) = \\hat{H}_0 +
         \\mathcal{\\hat{O}} \\cos(\\omega_d t)

    Parameters
    ----------
    H_0 : :obj:`.Qobj`
        A Hamiltonian or Liouvillian operator.

    c_ops : list
        A list of collapse operators.

    Op_t : :obj:`.Qobj`
        The the interaction operator which is multiplied by the cosine

    w_d : float, default: 1.0
        The frequency of the drive

    n_it : int, default: 3
        The number of iterations for the solver

    sparse : bool, default: False
        Solve for the steady state using sparse algorithms.

    solver : str, optional
        Solver to use when solving the linear system.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve"
          sparse solver by mkl.

        Extensions to qutip, such as qutip-tensorflow, may provide their own
        solvers. When ``H_0`` and ``c_ops`` use these data backends, see their
        documentation for the names and details of additional solvers they may
        provide.

    **kwargs:
        Extra options to pass to the linear system solver. See the
        documentation of the used solver in ``numpy.linalg`` or
        ``scipy.sparse.linalg`` to see what extra arguments are supported.

    Returns
    -------
    dm : qobj
        Steady state density matrix.

    Notes
    -----
    See: Sze Meng Tan,
    https://painterlab.caltech.edu/wp-content/uploads/2019/06/qe_quantum_optics_toolbox.pdf,
    Section (16)

    """

    L_0 = liouvillian(H_0, c_ops)
    L_m = 0.5 * liouvillian(Op_t)
    L_p = 0.5 * liouvillian(Op_t)
    # L_p and L_m correspond to the positive and negative
    # frequency terms respectively.
    # They are independent in the model, so we keep both names.
    Id = qeye_like(L_0)
    S = qzero_like(L_0)
    T = qzero_like(L_0)

    if isinstance(H_0.data, _data.CSR) and not sparse:
        L_0 = L_0.to("Dense")
        L_m = L_m.to("Dense")
        L_p = L_p.to("Dense")
        Id = Id.to("Dense")

    for n_i in np.arange(n_it, 0, -1):
        L = L_0 - 1j * n_i * w_d * Id + L_m @ S
        S.data = - _data.solve(L.data, L_p.data, solver, kwargs)
        L = L_0 + 1j * n_i * w_d * Id + L_p @ T
        T.data = - _data.solve(L.data, L_m.data, solver, kwargs)

    M_subs = L_0 + L_m @ S + L_p @ T
    return steadystate(M_subs, solver=solver, **kwargs)

# TODO: this function uses steadystate
def pseudo_inverse(L, rhoss=None, w=None, method='splu', *, use_rcm=False,
                   **kwargs):
    """
    Compute the pseudo inverse for a Liouvillian superoperator, optionally
    given its steady state density matrix (which will be computed if not
    given).

    Parameters
    ----------
    L : Qobj
        A Liouvillian superoperator for which to compute the pseudo inverse.

    rhoss : Qobj, optional
        A steadystate density matrix as Qobj instance, for the Liouvillian
        superoperator L.

    w : double, optional
        frequency at which to evaluate pseudo-inverse.  Can be zero for dense
        systems and large sparse systems. Small sparse systems can fail for
        zero frequencies.

    sparse : bool, optional
        Flag that indicate whether to use sparse or dense matrix methods when
        computing the pseudo inverse.

    method : str, optional
        Method used to compte matrix inverse.
        Choice are 'pinv' to use scipy's function of the same name, or a linear
        system solver.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab", "splu"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve",
          sparse solver by mkl.

        Extension to qutip, such as qutip-tensorflow, can use come with their
        own solver. When ``L`` use these data backends, see the corresponding
        libraries ``linalg`` for available solver.

    use_rcm : bool, default: False
        Use reverse Cuthill-Mckee reordering to minimize fill-in in the LU
        factorization of the Liouvillian.

    kwargs : dictionary
        Additional keyword arguments for setting parameters for solver methods.

    Returns
    -------
    R : Qobj
        Returns a Qobj instance representing the pseudo inverse of L.

    Notes
    -----
    In general the inverse of a sparse matrix will be dense.  If you
    are applying the inverse to a density matrix then it is better to
    cast the problem as an Ax=b type problem where the explicit calculation
    of the inverse is not required. See page 67 of "Electrons in
    nanostructures" C. Flindt, PhD Thesis available online:
    https://orbit.dtu.dk/en/publications/electrons-in-nanostructures-coherent-manipulation-and-counting-st

    Note also that the definition of the pseudo-inverse herein is different
    from numpys pinv() alone, as it includes pre and post projection onto
    the subspace defined by the projector Q.

    """
    if rhoss is None:
        rhoss = steadystate(L)

    sparse = kwargs.pop("sparse", False)
    if method == "direct":
        method = "splu" if sparse else "pinv"
    sparse_solvers = ["splu", "mkl_spsolve", "spilu"] # TODO: can be refactored too in a more maintainable structure
    dense_solvers = ["solve", "lstsq", "pinv"]
    if isinstance(L.data, (_data.CSR, _data.Dia)) and method in dense_solvers:
        L = L.to("dense")
    elif isinstance(L.data, _data.Dense) and method in sparse_solvers:
        L = L.to("csr")

    dtype = type(L.data)
    rhoss_vec = operator_to_vector(rhoss)

    tr_op = qeye_like(rhoss)
    tr_op_vec = operator_to_vector(tr_op)

    P = _data.kron(rhoss_vec.data, tr_op_vec.data.transpose(), dtype=dtype)
    I = _data.identity_like(P)
    Q = _data.sub(I, P)

    if w in [None, 0.0]:
        L += 1e-15j
    else:
        L += 1.0j * w

    use_rcm = use_rcm and isinstance(L.data, _data.CSR)

    if use_rcm: # If permutations should be used
        perm = scipy.sparse.csgraph.reverse_cuthill_mckee(L.data.as_scipy()) # TODO: the method is mentioned in the paper by P. Nation
        A = _data.permute.indices(L.data, perm, perm)
        Q = _data.permute.indices(Q, perm, perm, dtype=_data.CSR)
    else:
        A = L.data

    if method in ["pinv", "numpy", "scipy", "scipy2"]:
        # from scipy 1.7.0, they all use the same algorithm.
        LI = _data.Dense(scipy.linalg.pinv(A.to_array()), copy=False)
        LIQ = _data.matmul(LI, Q)
    elif method == "spilu":
        if not isinstance(A, (_data.CSR, _data.Dia)):
            warn("'spilu' method can only be used with sparse data.")
            A = _data.to(_data.CSR, A)
        ILU = scipy.sparse.linalg.spilu(A.as_scipy().tocsc(), **kwargs)
        LIQ = _data.Dense(ILU.solve(Q.to_array()))
    else:
        LIQ = _data.solve(A, Q, method, options=kwargs)

    R = _data.matmul(Q, LIQ)

    if use_rcm:
        rev_perm = np.argsort(perm)
        R = _data.permute.indices(R, rev_perm, rev_perm)

    return Qobj(R, dims=L._dims)


def _compute_precond(L, args):
    spilu_keys = {
        'permc_spec',
        'drop_tol',
        'diag_pivot_thresh',
        'fill_factor',
        'options',
    }
    ss_args = {
        key: args.pop(key)
        for key in spilu_keys
        if key in args
    }
    P = scipy.sparse.linalg.spilu(L.as_scipy().tocsc(), **ss_args)
    return scipy.sparse.linalg.LinearOperator(L.shape, matvec=P.solve)
