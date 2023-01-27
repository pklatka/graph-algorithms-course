# Patryk Klatka
"""
--- Opis algorytmu ---
Pierwszy sposób:
Zadanie jest pewną wariacją problemu plecakowego, w którym szukamy minimalnego kosztu zakupu przedmiotów o danej kategorii, tak aby ich suma wag była równa k, dodatkowo wykorzystując maksymalnie tylko jeden element z danej kategorii.

W naszym przypadku przedmiotami będą sumy zbiorów 1,2,...,x elementowych, gdzie elementy zbiorów to najmniejsze ceny występów + ceny ekwipunków odpowiednio dla ceny danego występu i artysty. Każde przedmioty są pogrupowane względem artystów, ponieważ artysta może wybrać tylko jedną wariację przedmiotu (może wykonać 1,2,...,x występów).

Funkcja dynamiczna:
F(k,i) - minimalny koszt zakupu przedmiotów o sumie wag k, spośród przedmiotów o indeksach od 1 do i.

F(0,i) = inf
F(k,0) = inf
F(k,i) = min(F(k,i-1), F(k-w[i], j-1) + c[i]), gdzie j to indeks pierwszego elementu z kategorii przedmiotu o indeksie i oraz k-w[i] > 0
F(k,i) = min(F(k,i-1), c[i]), gdzie k-w[i] <= 0

Rozwiązanie: min(F(K)), gdzie K to wymagana liczba występów.

Złożoność: O(n*m*k), gdzie n to liczba artystów, m - ilość możliwych do zorganizowania pokazów, a k to wymagana liczba występów.

Drugi sposób:
Ważną obserwacją jest fakt, że stawka bazowa za udział w każdym następnym pokazie jest wyższa niż w poprzednim. Sprawia, to że możemy stworzyć graf-ścieżkę dla każdego sztukmistrza. Każda ścieżka będzie miała maksymalnie m wierzchołków, gdzie krawędzie to wagi odpowiednio biorąc optymalnie 1, 2, 3, ... , m występów (optymalnie, czyli najtańsze występy). Przykładowo, gdy będziemy chcieli się dowiedzieć, jaki jest koszt dla aktora 3 występów biorąc optymalnie, to wystarczy dodać wagi trzech pierwszych krawędzi tej ścieżki. Gdy utworzymy grafy dla wszystkich sztukmistrzy, wystarczy k razy sprawdzić, która pierwsza krawędź ma najmniejszą wartość, dodać ją do rozwiązania i usunąć pobraną krawędź z grafu (możemy wspomóc się kopcem).

Złożoność: O(n*m*log(m)), gdzie n to liczba artystów, a m to ilość możliwych do zorganizowania pokazów.

UWAGA:
Algorytm jest poprawny tylko dla takich przypadków, gdy następny pokaz jest większy od poprzedniego, czyli tablica roznic jest rosnąca. W przeciwnym wypadku, algorytm nie jest poprawny, ponieważ nie zawsze wybierze optymalne rozwiązanie.
"""
import heapq
from collections import deque
from data import runtests


def my_solve_knapsack(N, M, K, base, wages, eq_cost):
    def knapsack(costs, weights, K, first_element_in_category):
        n = len(costs)

        F = [[0 for _ in range(n + 1)] for _ in range(K + 1)]

        for i in range(n + 1):
            F[0][i] = float('inf')

        for k in range(K + 1):
            F[k][0] = float('inf')

        for k in range(1, K + 1):
            for i in range(1, n + 1):
                item_index = i-1
                F[k][i] = F[k][i - 1]

                if k - weights[item_index] > 0:
                    F[k][i] = min(F[k][i], F[k - weights[item_index]]
                                  [first_element_in_category[item_index] - 1] + costs[item_index])
                else:
                    F[k][i] = min(F[k][i], costs[item_index])

        return min(F[K])

    costs = []
    weights = []
    first_element_in_category = []

    # Sort shows
    for i in range(len(wages)):
        wage = wages[i]
        q = []
        for j in range(len(wage)):
            s = wage[j]
            heapq.heappush(q, (s[1] + eq_cost[s[0] - 1], s[0]))
        wages[i] = q

    iterator = 1
    for i in range(len(base)):
        x = base[i]
        max_to_take = min(len(x), len(wages[i]))
        s = 0
        for v in range(max_to_take):
            weights.append(v+1)
            s += heapq.heappop(wages[i])[0]
            costs.append(x[v] + s)

            first_element_in_category.append(iterator)
        iterator += max_to_take

    return knapsack(costs, weights, K, first_element_in_category)


def my_solve(N, M, K, base, wages, eq_cost):
    # Sort shows: O(n*m*log(m)) -> O(n*m), because we are building a heap
    for i in range(len(wages)):
        wage = wages[i]
        q = []
        for j in range(len(wage)):
            s = wage[j]
            heapq.heappush(q, (s[1] + eq_cost[s[0] - 1], s[0]))
        wages[i] = q

    actors = []

    # O(n*m*log(m)) -> O(n*m)
    for actor in range(N):
        max_to_take = min(len(base[actor]), len(wages[actor]))
        actors.append(deque(maxlen=max_to_take))
        base_before = 0
        for i in range(max_to_take):
            play = heapq.heappop(wages[actor])[0]
            actors[actor].append(play + (base[actor][i] - base_before))
            base_before = base[actor][i]

    q = []
    for i in range(len(actors)):
        heapq.heappush(q, (actors[i].popleft(), i))

    ans = 0
    # O(k*log(n))
    for i in range(K):
        value, index = heapq.heappop(q)
        ans += value

        if actors[index]:
            heapq.heappush(q, (actors[index].popleft(), index))

    return ans


runtests(my_solve)
