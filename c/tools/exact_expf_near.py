"""exact_expf_near.py <file>: the near_cases[] rows of tests/test_exact_expf.c.

<file> is the output of `tests/test_exact_expf --near`: the floats whose exp, by the C
library's double exp, lies within 2^-49 of the middle between two floats, one 0x........u per
line. For each one this prints {x bits, exp(x) correctly rounded to float bits}, worked out with
the standard library only: decimal at 60 digits, then exact rounding with Fraction (ties cannot
occur: the exp of a nonzero float is irrational)."""
import struct
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60
FLT_MAX = Fraction(struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0])


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def round_f32(v):
    """v > 0 exact Fraction -> bits of the nearest float32, ties to even"""
    if v >= FLT_MAX + Fraction(2) ** 103:
        return 0x7F800000
    e = 0
    while Fraction(2) ** (e + 1) <= v:
        e += 1
    while Fraction(2) ** e > v:
        e -= 1
    ulp = Fraction(2) ** max(e - 23, -149)  # v in [2^e, 2^(e+1)): 24 significant bits, or the subnormal grid
    q = v / ulp
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    r = float(n * ulp)  # exact: n * ulp is a float32 value
    return struct.unpack("<I", struct.pack("<f", r))[0]


for line in open(sys.argv[1]):
    xb = int(line.strip().rstrip("u"), 16)
    x = f32(xb)
    y = Decimal(x).exp()  # Decimal(float) is exact
    assert Decimal(x) != 0
    yb = round_f32(Fraction(y))
    # 60 digits leave the rounding unambiguous: the value is nowhere near 1e-50 of a float midpoint
    lo, hi = Fraction(y) * (1 - Fraction(1, 10**50)), Fraction(y) * (1 + Fraction(1, 10**50))
    assert round_f32(lo) == round_f32(hi) == yb, hex(xb)
    print(f"    {{0x{xb:08x}u, 0x{yb:08x}u}}, /* exp({x.hex()}) */")
