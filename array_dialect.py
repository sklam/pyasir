import numpy as np
import operator

def array_map(func, arg):
    return np.asarray([func(x) for x in arg])

def array_transpose(a):
    return a.T

def array_reduce(func, arg):
    return sum(arg)

def mul_pair(pairs):
    return [a * b for a, b in pairs]

def zip_array(A, B):
    return list(zip(A, B))


scalar_add = operator.add

# https://github.com/rise-lang/shine/blob/main/docs/exploration/tutorial.md
# val mm =
#   fun(ArrayType(N, ArrayType(N, f32)))(a =>
#     fun(ArrayType(N, ArrayType(N, f32)))(b =>
#       a |> map(fun(ak =>
#         b |> transpose |> map(fun(bk =>
#           zip(ak)(bk) |>
#             map(fun(x => fst(x) * snd(x))) |>
#             reduce(add)(lf32(0.0f)) )) )) ))
def matmul(A, B):
    def proc_A(arow):
        def proc_B(bcol):
            pair = zip_array(arow, bcol)
            prod = mul_pair(pair)
            return array_reduce(scalar_add, prod)
        return array_map(proc_B, array_transpose(B))
    return array_map(proc_A, A)


def array_function(*out_shape):
    def decor(fn):
        def wrapped(*args):
            out = np.zeros(out_shape)
            for index in np.ndindex(out.shape):
                res = fn(index, *args)
                out[index] = res
            return out
        return wrapped
    return decor

def matmul_dialect(A, B):
    @array_function(A.shape[0], B.shape[1])
    def do(index, A, B):
        """
        for each i j
           Out[i, j] = sum(A[i] * B[j])
        """
        i, j = index
        arow = A[i]
        bcol = B[j]
        pair = zip_array(arow, bcol)
        prod = mul_pair(pair)
        return array_reduce(scalar_add, prod)

    return do(A, array_transpose(B))


def test():
    A = np.arange(3 * 4).reshape(3, 4)
    B = np.arange(4 * 3).reshape(4, 3)

    print(A)
    print(B)

    C = matmul(A, B)

    print(C)
    Cexp = A @ B
    print(Cexp)


    Cdia = matmul_dialect(A, B)
    print(Cdia)





test()