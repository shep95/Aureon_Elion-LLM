"""Shamir secret sharing over GF(256) — nomad key ceremony pattern."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

_EXP = bytearray(512)
_LOG = bytearray(256)


def _gf_multiply(a: int, b: int) -> int:
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def _init_gf256() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x = _gf_multiply(x, 3)
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_init_gf256()


def _gf_add(a: int, b: int) -> int:
    return a ^ b


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[(_LOG[a] + _LOG[b]) % 255]


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ValueError("GF division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b] + 255) % 255]


def _eval_poly(coeffs: list[int], x: int) -> int:
    result = 0
    for coeff in reversed(coeffs):
        result = _gf_add(_gf_mul(result, x), coeff)
    return result


@dataclass
class ShamirShare:
    index: int
    data: bytes


def split_secret(secret: bytes, threshold: int, shares: int) -> list[ShamirShare]:
    if threshold < 2:
        raise ValueError("Threshold must be >= 2")
    if shares < threshold:
        raise ValueError("Shares must be >= threshold")
    result = [ShamirShare(index=x, data=bytearray(len(secret))) for x in range(1, shares + 1)]
    for byte_idx in range(len(secret)):
        coeffs = [secret[byte_idx]] + [secrets.randbelow(256) for _ in range(threshold - 1)]
        for x in range(1, shares + 1):
            result[x - 1].data[byte_idx] = _eval_poly(coeffs, x)
    return [ShamirShare(index=s.index, data=bytes(s.data)) for s in result]


def combine_shares(shares: list[ShamirShare]) -> bytes:
    if not shares:
        raise ValueError("No shares provided")
    length = len(shares[0].data)
    out = bytearray(length)
    for byte_idx in range(length):
        value = 0
        for i, share_i in enumerate(shares):
            basis = 1
            xi = share_i.index
            yi = share_i.data[byte_idx]
            for j, share_j in enumerate(shares):
                if i == j:
                    continue
                xj = share_j.index
                basis = _gf_mul(basis, _gf_div(xj, _gf_add(xi, xj)))
            value = _gf_add(value, _gf_mul(yi, basis))
        out[byte_idx] = value
    return bytes(out)
