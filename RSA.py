import random
import secrets
import sys
import math
# from random import randrange


def multiply(x, y):
    _CUTOFF = 1536
    if x.bit_length() <= _CUTOFF or y.bit_length() <= _CUTOFF:  # Base case
        return x * y
    else:
        n = max(x.bit_length(), y.bit_length())
        half = (n + 32) // 64 * 32
        mask = (1 << half) - 1
        xlow = x & mask
        ylow = y & mask
        xhigh = x >> half
        yhigh = y >> half

        a = multiply(xhigh, yhigh)
        b = multiply(xlow + xhigh, ylow + yhigh)
        c = multiply(xlow, ylow)
        d = b - a - c
        return (((a << half) + d) << half) + c


def inverse(e, drawrange):
    x = 0
    y = 1
    l_x = 1
    l_y = 0
    orig_e = e  # e i drawrange żeby usunć negatywne wartości
    orig_drawrange = drawrange
    while drawrange != 0:
        q = e // drawrange
        (e, drawrange) = (drawrange, e % drawrange)
        (x, l_x) = ((l_x - (q * x)), x)
        (y, l_y) = ((l_y - (q * y)), y)
    if l_x < 0:
        l_x += orig_drawrange
    if l_y < 0:
        l_y += orig_e
    return l_x


def inverse2(e, drawrange):
    for i in range(1, drawrange):
        if ((e % drawrange) * (i % drawrange)) % drawrange == 1:
            return i
    return None


def test(a, r, d, num):
    """
    :param a: podstawa potęgi
    :param r: wykłądnik
    :param d: współczynnik
    :param num: liczba
    :return:
    """
    x = pow(a, d, num)  # x = a^d mod num
    if x == 1 or x == num - 1:
        return True
    for i in range(1, r - 1):
        x = pow(x, 2, num)
        if x == num - 1:
            return True
    return False


def rabinMiller(num, iterations=100):
    """
    :param num: liczba do analizy
    :param iterations: liczba iteracji
    :return: True - prawdopodobnie pierwsza; False - złożona
    """
    if num == 2:  # wstępne przetwarzanie oczywistych liczb
        return True
    if num % 2 == 0:
        return False

    r = 0  # budowanie liczby do testu jako num = 2**r*d+1
    d = num - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    for i in range(1, iterations):  # test
        a = 0
        while a == 0:
            a = secrets.randbelow(num - 1)
        if not test(a, r, d, num):
            return False
    return True


def gcd(a, b):
    """
    :param a: liczba do porównania
    :param b: liczba do porównania
    :return: 1 gdy względnie pierwsze
    """
    while b != 0:
        a, b = b, a % b
    return a


def choose(size):
    """
    :param size: potęga 2 - górna granica losowania
    :return: wylosowana i przetestowana liczba
    """
    for _ in reversed(range(int(100 * (math.log(size, 2) + 1)))):  # liczba prób
        n = secrets.randbits(size)
        while n <= 2 ** (size - 1):
            n = secrets.randbits(size)
        if rabinMiller(n):
            return n


def choose_prime_numbers(size):
    # wybór liczb pierwszych
    p = choose(size)
    q = choose(size)
    while p == q:
        print('p and q are equal. Recalculate.')
        p = choose(size)
        q = choose(size)

    # wyznaczenie num
    num = multiply(p, q)

    # zakres losowania
    drawrange = multiply((p-1),(q-1))  # Funkcja λ (lambda) Carmichaela
    g = 0
    while g != 1 or e == 0:
        # względnie pierwsza e
        e = secrets.randbelow(drawrange)
        # czy względnie pierwsza (Euclid Algorithm)
        g = gcd(e, drawrange)

    # część prywatna
    d = inverse(e, drawrange)  # e^-1 mod drawrange
    print(d)
    # d = inverse2(e, drawrange)  # e^-1 mod drawrange
    # print(d)
    # Zwróć klucze odpowiednio publiczny i prywatny i parę liczb
    return (e, num), (d, num), (p, q)


def RSA():  # to albo od razu choose_prime_numbers
    public, private, prime = choose_prime_numbers(100) # argument to potęga 2 odpowiadająca górnej granicy liczb
    return public, private, prime


public, private, prime = choose_prime_numbers(1000)
print(public, private, prime)
