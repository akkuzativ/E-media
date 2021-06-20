import random
import secrets
import sys
import math
from random import randrange


def test(a, r, d, num):
    x = pow(a, d, num)
    for i in range(1, r - 1):
        if x == num - 1:
            return True
        x = pow(x, 2, num)
    return x == 1 or x == num - 1


def rabinMiller(num, iterations=100):
    """
    :param num: liczba
    :param iterations: liczba iteracji
    :return: True - prawdopodobnie pierwsza; False - złożona
    """
    if num == 2:
        return True
    if not num & 1:
        return False

    r = 0  # potęga 2 w num = 2**r*d+1
    d = num - 1  # ----------------^

    while d % 2 == 0:
        d >>= 1  # dziel przez 2**1
        r += 1

    for i in range(1, iterations):
        a = randrange(2, num - 1)
        if not test(a, r, d, num):
            return False
    return True


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


def inverse(a, b):
    x = 0
    y = 1
    lx = 1
    ly = 0
    oa = a  # Remember original a/b to remove
    ob = b  # negative values from return results
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lx) = ((lx - (q * x)), x)
        (y, ly) = ((ly - (q * y)), y)
    if lx < 0:
        lx += ob  # If neg wrap modulo orignal b
    if ly < 0:
        ly += oa  # If neg wrap modulo orignal a
    # return a , lx, ly  # Return only positive values
    return lx


# def is_prime(num):
#     # lowPrimes is all primes (sans 2, which is covered by the bitwise and operator)
#     # under 1000. taking num modulo each lowPrime allows us to remove a huge chunk
#     # of composite numbers from our potential pool without resorting to Rabin-Miller
#     lowPrimes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
#         , 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179
#         , 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269
#         , 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367
#         , 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461
#         , 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571
#         , 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661
#         , 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773
#         , 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883
#         , 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
#     if (num >= 3):
#         if (num & 1 != 0):
#             for p in lowPrimes:
#                 if (num == p):
#                     return True
#                 if (num % p == 0):
#                     return False
#             return rabinMiller(num)
#     return False


def choose(size):
    for _ in reversed(range(int(100 * (math.log(size, 2) + 1)))):  # liczba prób
        n = secrets.randbelow(2 ** size)
        while n <= 2 ** (size - 1):
            n = secrets.randbelow(2 ** size)
        # if is_prime(num) == True:
        if rabinMiller(n):
            return n


def choose_prime_numbers(size):
    # wybór liczb pierwszych
    p = choose(size)
    q = choose(size)
    while p == q:
        print('p and q are equal. Recalculate q.')
        q = choose(size)

    # wyznaczenie num
    # num = multiply(p, q)
    n = p * q

    # zakres losowania
    # drawrange = multiply((p-1),(q-1))
    drawrange = (p - 1) * (q - 1)
    g = 0
    while g != 1:
        # względnie pierwsza e
        e = random.randrange(1, drawrange)
        # czy względnie pierwsza (Euclid'r Algorithm)
        g = gcd(e, drawrange)

    # część prywatna
    d = inverse(e, drawrange)

    # Zwróć klucze odpowiednio publiczny i prywatny i paręliczb
    return (e, n), (d, n), (p, q)


def fun():
    public, private, prime = choose_prime_numbers(100) # argument to potęga 2 odpowiadająca górnej granicy liczb


choose_prime_numbers(100)
