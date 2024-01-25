import numpy as np
import openmdao.api as om
from math import isclose


def parallel_check_partials(prob: om.Problem, out_slice, rev_slice):
    # preparation
    sizes = np.array(
        [
            prob.model._doutputs.asarray()[out_slice].size,
            prob.model._dresiduals.asarray()[rev_slice].size,
        ],
        dtype=np.int32,
    )
    comm = prob.comm
    global_sizes = np.zeros((2, comm.size), np.int32)
    comm.Allgather(sizes, global_sizes)
    fwd_derivs = {}
    rev_derivs = {}
    # fwd iteration
    for i1 in range(comm.size):
        fwd_derivs[(comm.rank, i1)] = np.zeros((global_sizes[0][i1], sizes[1]))
        for i2 in range(global_sizes[0][i1]):
            prob.model._doutputs.set_val(0)
            prob.model._dresiduals.set_val(0)
            if comm.rank == i1:
                prob.model._doutputs.asarray()[out_slice[i2]] = 1.0
            prob.model.run_apply_linear(mode="fwd")
            fwd_derivs[(comm.rank, i1)][:, i2] = prob.model._dresiduals.asarray()[
                rev_slice
            ]
    # rev iteration
    for i1 in range(comm.size):
        rev_derivs[(i1, comm.rank)] = np.zeros((global_sizes[1][i1], sizes[0]))
        for i2 in range(global_sizes[1][i1]):
            prob.model._doutputs.set_val(0)
            prob.model._dresiduals.set_val(0)
            if comm.rank == i1:
                prob.model._dresiduals.asarray()[rev_slice[i2]] = 1.0
            prob.model.run_apply_linear(mode="rev")
            rev_derivs[(i1, comm.rank)][:, i2] = prob.model._doutputs.asarray()[
                out_slice
            ]

    # exchange

    for i1 in range(comm.size):
        if i1 < comm.rank:
            comm.Send(rev_derivs[(i1, comm.rank)], dest=i1, tag=comm.rank)
            rev_derivs[(comm.rank, i1)] = np.zeros((sizes[1], global_sizes[0][i1]))
            comm.Recv(rev_derivs[(comm.rank, i1)], source=i1, tag=i1)
        elif i1 > comm.rank:
            rev_derivs[(comm.rank, i1)] = np.zeros((sizes[1], global_sizes[0][i1]))
            comm.Recv(rev_derivs[(comm.rank, i1)], source=i1, tag=i1)
            comm.Send(rev_derivs[(i1, comm.rank)], dest=i1, tag=comm.rank)

    # compare
    truth_array = np.zeros(comm.size, bool)
    for i1 in range(comm.size):
        truth_array[i1] = True
        for i in range(sizes[1]):
            for j in range(global_sizes[0][i1]):
                truth_array[i1] = truth_array[i1] and isclose(
                    fwd_derivs[(comm.rank, i1)][i, j],
                    rev_derivs[(comm.rank, i1)][j, i],
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
        if not truth_array[i1]:
            print(comm.rank, i1, fwd_derivs[(comm.rank, i1)])
            print(comm.rank, i1, rev_derivs[(comm.rank, i1)])
            print(comm.rank, i1, rev_derivs[(i1, comm.rank)])
    return truth_array
