from llpyfe.types import Int64


def do_sum(n: Int64) -> Int64:
    c = 0
    for i in range(n):
        # print('i =', i)
        c += i
    return c


def main():
    print("A")
    r = do_sum(10)
    print("r =", r)


