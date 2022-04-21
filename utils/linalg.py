import torch
import cupy
from cupyx.scipy.sparse.linalg import lsmr


@torch.no_grad()
def closed_form_solver(A, B):
    if B.shape[0] == 1:
        X = B / A[0, 0]
    else:
        # NOTE: for safety, compute matrix inverse on CPU
        X = torch.inverse(A.cpu()).to(A.device) @ B
    return X


@torch.no_grad()
def lsmr_cupy_solver(A, B):
    B = B - A.sum(dim=1)
    if B.shape[0] == 1:
        X = B / A[0, 0]
    else:
        CU_A = cupy.asarray(A.cpu().numpy())
        CU_B = cupy.asarray(B.cpu().numpy())
        solution = lsmr(CU_A, CU_B, damp=1)
        X = cupy.asnumpy(solution[0])
        X = torch.from_numpy(X).to(A.device)
    X = X + 1
    return X, solution[1] < 3
