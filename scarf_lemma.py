import numpy as np


def scarf_lemma(A, b, C):
    n = A.shape[0]
    a = np.arange(n)
    c = np.append(np.arange(1, n), n+np.argmax(C[0, n:]))

    j = np.setdiff1d(c, a, assume_unique=True)[0]

    step = 1
    while True:
        print(f"Step {step}: a = {a+1}, c = {c+1}")  # index 0 スタートを 1 スタートに直して出力
        A, a, b, j = cardinal_pivot(A, a, b, j)
        if (a == c).all():
            break

        c, j = ordinal_pivot(C, c, j)
        if (a == c).all():
            break
        step += 1

    # index 0 スタートを 1 スタートに直してreturn
    return a+1, c+1


def cardinal_pivot(A, a, b, i):
    m, n = A.shape
    ratio = np.array([])

    for row in range(m):
        if A[row, i] > 0.01:
            if ratio.size == 0:
                ratio = np.array([b[row] / A[row, i], row])
            else:
                ratio = np.vstack([ratio, np.array([b[row] / A[row, i], row])])
    if ratio.size == 0:
        raise ValueError("i can not be a basis column.")
    elif ratio.size == 2:
         pivot_row = ratio[1].astype(int)
    else:
        idx = np.argmin(ratio, axis=0)[0]
        pivot_row = ratio[idx, 1].astype(int)
    basis = A[:, a]
    basis_col = np.argmax(basis[pivot_row, :])
    pivot = a[basis_col]

    a[a==pivot] = i
    a.sort()

    A_new = np.zeros(shape=(m,n))
    b_new = np.zeros(shape=b.shape)

    A_new[pivot_row, :] = A[pivot_row, : ] / A[pivot_row, i]
    b_new[pivot_row] = b[pivot_row] / A[pivot_row, i]

    for k in range(m):
        if k != pivot_row:
            A_new[k, :] = A[k, :] - A[k, i] * A_new[pivot_row, :]
            b_new[k] = b[k] - A[k, i] * b_new[pivot_row]

    return [A_new, a, b_new, pivot]


def ordinal_pivot(C, c, j):
    n = C.shape[1]
    c_bar = np.setdiff1d(np.arange(n), c, assume_unique=True)
    b = C[:, c_bar]

    basis = C[:, c]
    c_remains = np.setdiff1d(c, j, assume_unique=True)
    basis2 = C[:, c_remains]

    original_argmins = np.argmin(basis, axis=1)
    original_argmins = c[original_argmins]

    new_mins = np.min(basis2, axis=1)
    new_argmins = np.argmin(basis2, axis=1)
    new_argmins = c_remains[new_argmins]

    unique, counts = np.unique(new_argmins, return_counts=True)
    r = unique[counts >= 2]
    if r.size > 1:
        raise ValueError("Two columns with two row minimizer.")
    r = r[0]

    idx = np.where(original_argmins == r)[0][0]

    row_of_idx = b[idx, :]
    b = np.delete(b, idx, axis=0)
    new_mins = np.delete(new_mins, idx, axis=0)
    all_mins = np.tile(new_mins.reshape((new_mins.size, 1)), (1, b.shape[1]))
    cols = np.where((b > all_mins).all(axis=0))[0]

    new_col = cols[np.argmax(row_of_idx[cols])]
    new_c = c_bar[new_col]
    c[c==j] = new_c
    c.sort()

    return c, new_c
