from testy import *


def lcg(seed):
    a = 1103515245
    a = 69069
    c = 12345
    m = 2 ** 31
    while True:
        yield seed ^ (seed >> 16)
        seed = (a * seed + c) % m


generator = lcg(314159265)
def myrand(a, b): return a + next(generator) % (b - 1)


def random_cost_list(cap, step_min, step_max):
    c = 0
    v = 0
    a = 0
    costs = []
    for _ in range(cap):
        a = myrand(step_min, step_max)
        v += a
        c += v
        costs.append(c)
    return costs


def make_dense(N, M, K, cap_min, cap_max, step_min, step_max, cmin, cmax, eqmin, eqmax):

    base = [random_cost_list(myrand(cap_min, cap_max),
                             step_min, step_max) for _ in range(N)]

    wages = []
    for _ in range(N):
        current = []
        for j in range(1, M + 1):
            cost = myrand(cmin, cmax)
            current.append((j, cost))
        wages.append(current)

    eq_cost = [myrand(eqmin, eqmax) for _ in range(M)]

    return [N, M, K, base, wages, eq_cost]


def make_sparse(N, M, K, conn, cap_min, cap_max, step_min, step_max, cmin, cmax, eqmin, eqmax):
    def flip(): return myrand(0, 1000) / 1000.0 < conn

    base = [random_cost_list(myrand(cap_min, cap_max),
                             step_min, step_max) for _ in range(N)]

    wages = []
    for _ in range(N):
        current = []
        for j in range(1, M + 1):
            if flip():
                cost = myrand(cmin, cmax)
                current.append((j, cost))
        wages.append(current)

    eq_cost = [myrand(eqmin, eqmax) for _ in range(M)]

    return [N, M, K, base, wages, eq_cost]


problems = [
    # Przykład z opisu zadania
    {"arg": [3, 4, 5,
             [(3, 8), (1, 4, 11), (5, 13)],   # wynagrodzenia bazowe
             [
                 [(1, 15), (2, 7)],             # pokazy pierwszego artysty
                 [(2, 11), (3, 17)],            # pokazy drugiego artysty
                 [(1, 25), (2, 17), (4, 31)]    # pokazy trzeciego artysty
             ],
             [5, 3, 10, 7]                    # ceny ekwipunku dla pokazów
             ],
     "hint": 108
     },
    {"arg": [5, 5, 5,
             [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)],
             [
                 [(3, 10), (1, 180), (2, 341)],
                 [(1, 10), (2, 300), (5, 430)],
                 [(2, 10), (5, 300), (3, 400)],
                 [(4, 10), (3, 250), (1, 500)],
                 [(5, 10), (4, 234), (3, 450)],
             ],
             [5, 6, 7, 8, 9]
             ],
        "hint": 90
     },
    {"arg": [4, 6, 5,
             [(100, 1000), (100, 1000), (4, 16, 64, 128, 400), (100, 1000)],
             [
                 [(1, 10), (2, 15)],
                 [(3, 10), (4, 15)],
                 [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10)],
                 [(5, 10), (6, 10)],
             ],
             [20, 16, 12, 10, 8, 7]
             ],
        "hint": 316
     },
    {"arg": [3, 3, 1,
             [(3,), (5,), (6,)],
             [
                 [(1, 5), (2, 5), (3, 4)],
                 [(1, 4), (2, 4), (3, 4)],
                 [(1, 3), (2, 8), (3, 2)],
             ],
             [5, 6, 9],
             ],
        "hint": 13,
     },
    {"arg": make_dense(5, 6, 15, 3, 5, 4, 7, 10, 30, 5, 20),
     "hint": 602
     },
    {"arg": make_dense(40, 30, 300, 5, 10, 4, 10, 10, 30, 5, 18),
     "hint": 17869
     },
    {"arg": make_sparse(20, 30, 13, 0.2, 5, 10, 4, 10, 10, 30, 5, 18),
     "hint": 422
     },
    {"arg": make_sparse(200, 100, 280, 0.1, 5, 10, 4, 10, 10, 30, 5, 18),
     "hint": 9271
     },
]


def printarg(N, M, K, base, wages, eq_cost):
    print(f"{N} sztukmistrzów, {M} pokazów, {K} wystąpień")
    print(f"Bazowe wynagrodzenia: {limit(base, 120)}")
    print(f"Za występ: {limit(wages, 120)}")
    print(f"Za wyosażenie: {limit(eq_cost, 120)}")


def printhint(hint):
    print("Wynik: {}".format(hint))


def printsol(sol):
    print("Uzyskany wynik: {}".format(sol))


def check(N, M, K, base, wages, eq_cost, hint, sol):
    if hint == sol:
        print("Test zaliczony")
        return True
    else:
        print("NIEZALICZONY!")
        return False


def runtests(f):
    internal_runtests(printarg, printhint, printsol, check, problems, f)
