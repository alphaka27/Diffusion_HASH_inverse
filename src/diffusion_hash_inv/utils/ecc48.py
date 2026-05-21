"""
48-bit [n=48, k=8] Error Correction Codecs for the RGB pair encoding scheme.

All three codecs encode one byte (8-bit payload) into a 48-bit codeword that is
packed into two RGB pixel values as defined in ``Encoding Method.md``:

    C[0:8]  → RGB_1.R
    C[8:16] → RGB_1.G
    C[16:24]→ RGB_1.B
    C[24:32]→ RGB_2.R
    C[32:40]→ RGB_2.G
    C[40:48]→ RGB_2.B

Codec comparison
----------------
+------------------+----------+------------------+----------------------------+
| Codec            | Code     | Max correctable  | Notes                      |
+==================+==========+==================+============================+
| Golay24Dual      | [48,8]   | 6 bit-errors     | 2× Extended Golay(24,12)   |
| RS48             | [48,8]   | 2 byte-errors    | RS(6,1)/GF(2^8), ≤16 bits  |
| BCH48            | [48,8]   | 7 bit-errors     | BCH[63,24,15] shortened    |
+------------------+----------+------------------+----------------------------+

All decoders return a ``DecodeResult`` that includes a ``confidence`` score
(0.0 – no confidence … 1.0 – perfect) alongside the recovered payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    """Decode outcome with reliability metadata."""

    valid: bool
    """True if the payload was successfully recovered (possibly after correction)."""

    payload: int | None
    """Recovered byte value (0–255), or None if uncorrectable."""

    confidence: float
    """Reliability estimate in [0.0, 1.0].  1.0 = no errors detected."""

    errors_corrected: int
    """Number of bit-errors (Golay24Dual, BCH48) or byte-errors (RS48) corrected."""

    method: str
    """One of ``'golay24-dual'``, ``'rs48'``, ``'bch48'``."""

    uncorrectable: bool
    """True when the error pattern exceeds the codec's correction capability."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Method-specific diagnostics (syndromes, syndrome weight, etc.)."""


# ---------------------------------------------------------------------------
# GF(2^8) field arithmetic  (used by RS48)
# ---------------------------------------------------------------------------

class _GF256:
    """GF(2^8) arithmetic with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D), α=2."""

    _EXP: list[int] = []
    _LOG: list[int] = []
    _PRIM = 0x11D   # x^8 + x^4 + x^3 + x^2 + 1 (α=2 is primitive)

    @classmethod
    def _ensure_tables(cls) -> None:
        if cls._EXP:
            return
        exp: list[int] = [0] * 512
        log: list[int] = [0] * 256
        a = 1
        for i in range(255):
            exp[i] = a
            log[a] = i
            a <<= 1
            if a & 0x100:
                a ^= cls._PRIM
            a &= 0xFF
        for i in range(255, 512):
            exp[i] = exp[i - 255]
        cls._EXP = exp
        cls._LOG = log

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        cls._ensure_tables()
        return cls._EXP[cls._LOG[a] + cls._LOG[b]]

    @classmethod
    def div(cls, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("GF256 division by zero")
        cls._ensure_tables()
        return cls._EXP[(cls._LOG[a] - cls._LOG[b]) % 255]

    @classmethod
    def pow(cls, a: int, n: int) -> int:
        if n == 0:
            return 1
        if a == 0:
            return 0
        cls._ensure_tables()
        return cls._EXP[(cls._LOG[a] * n) % 255]

    @classmethod
    def inv(cls, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF256 inverse of zero")
        cls._ensure_tables()
        return cls._EXP[255 - cls._LOG[a]]


# ---------------------------------------------------------------------------
# GF(2^6) field arithmetic  (used by BCH48)
# ---------------------------------------------------------------------------

class _GF64:
    """GF(2^6) arithmetic with primitive polynomial x^6 + x + 1 (0x43)."""

    ORDER = 63
    _PRIM = 0x43   # leading bit dropped during shift-based reduction
    _REDUCE = 0x03  # = (0x43 ^ 0x40) = x + 1  (reduction term)

    _EXP: list[int] = []
    _LOG: list[int] = []

    @classmethod
    def _ensure_tables(cls) -> None:
        if cls._EXP:
            return
        exp: list[int] = [0] * (cls.ORDER * 2)
        log: list[int] = [0] * (cls.ORDER + 1)
        a = 1
        for i in range(cls.ORDER):
            exp[i] = a
            log[a] = i
            a <<= 1
            if a & 0x40:
                a ^= cls._PRIM
            a &= 0x3F
        for i in range(cls.ORDER, cls.ORDER * 2):
            exp[i] = exp[i - cls.ORDER]
        cls._EXP = exp
        cls._LOG = log

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        cls._ensure_tables()
        return cls._EXP[(cls._LOG[a] + cls._LOG[b]) % cls.ORDER]

    @classmethod
    def pow(cls, a: int, n: int) -> int:
        if n == 0:
            return 1
        if a == 0:
            return 0
        cls._ensure_tables()
        return cls._EXP[(cls._LOG[a] * n) % cls.ORDER]

    @classmethod
    def inv(cls, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF64 inverse of zero")
        cls._ensure_tables()
        return cls._EXP[cls.ORDER - cls._LOG[a]]

    @classmethod
    def alpha_pow(cls, n: int) -> int:
        """Return α^n as a GF(2^6) element."""
        cls._ensure_tables()
        return cls._EXP[n % cls.ORDER]


# ---------------------------------------------------------------------------
# Berlekamp-Massey for both GF(2^6) and GF(2^8)
# ---------------------------------------------------------------------------

def _berlekamp_massey(syndromes: list[int], gf_mul, gf_inv) -> list[int]:
    """
    Berlekamp-Massey algorithm.

    Returns the error-locator polynomial Λ as a list of field elements
    [Λ_0, Λ_1, ..., Λ_t] where Λ_0 = 1.
    """
    n = len(syndromes)
    C: list[int] = [1]
    B: list[int] = [1]
    L = 0
    m = 1
    b = 1

    for step in range(n):
        d = syndromes[step]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= gf_mul(C[i], syndromes[step - i])

        if d == 0:
            m += 1
        elif 2 * L <= step:
            T = list(C)
            factor = gf_mul(d, gf_inv(b))
            while len(C) < len(B) + m:
                C.append(0)
            for i, coeff in enumerate(B):
                C[i + m] ^= gf_mul(factor, coeff)
            L = step + 1 - L
            B = T
            b = d
            m = 1
        else:
            factor = gf_mul(d, gf_inv(b))
            while len(C) < len(B) + m:
                C.append(0)
            for i, coeff in enumerate(B):
                C[i + m] ^= gf_mul(factor, coeff)
            m += 1

    return C


# ===========================================================================
# Codec 1: Golay24Dual
# ===========================================================================

class Golay24DualCodec:
    """
    2× Extended Golay(24,12) codec.

    The 8-bit payload is encoded with the Extended Golay code to produce a
    24-bit codeword; the same codeword is stored twice, giving 48 bits.
    Each half can independently correct up to 3 bit-errors, so the pair can
    survive any pattern with ≤ 6 total errors spread across both halves, or
    up to 3 errors concentrated in one half.

    Decode confidence
    -----------------
    - Both halves valid & agree,  0 errors total : 1.00
    - Both halves valid & agree,  e errors total : 1.00 − 0.10·e
    - Only one half valid,        0 errors       : 0.75
    - Only one half valid,        e errors       : 0.75 − 0.10·e
    - Both valid but disagree     (lower-error wins): 0.20
    - Both uncorrectable                          : 0.00
    """

    METHOD = "golay24-dual"
    CODEWORD_BITS = 48
    PIXELS_PER_BYTE = 2
    DATA_BITS = 8

    # Golay(23) parameters
    _G23_GEN = 0b101011100011
    _G23_PARITY_BITS = 11
    _G23_CODE_BITS = 23
    _G23_MSG_BITS = 12
    _G24_CODE_BITS = 24
    _G23_MAX_CORRECT = 3

    _syndrome_cache: dict[int, int] | None = None

    # ------------------------------------------------------------------
    # Golay(23/24) primitives  (mirrors Byte2RGB internals)
    # ------------------------------------------------------------------

    @classmethod
    def _poly_mod(cls, value: int, gen: int) -> int:
        deg = gen.bit_length() - 1
        while value and value.bit_length() - 1 >= deg:
            value ^= gen << (value.bit_length() - 1 - deg)
        return value

    @classmethod
    def _syndrome(cls, code23: int) -> int:
        return cls._poly_mod(code23 & ((1 << cls._G23_CODE_BITS) - 1), cls._G23_GEN)

    @classmethod
    def _syndrome_table(cls) -> dict[int, int]:
        if cls._syndrome_cache is not None:
            return cls._syndrome_cache
        table: dict[int, int] = {0: 0}
        for weight in range(1, cls._G23_MAX_CORRECT + 1):
            for combo in combinations(range(cls._G23_CODE_BITS), weight):
                pat = sum(1 << (cls._G23_CODE_BITS - 1 - p) for p in combo)
                table.setdefault(cls._syndrome(pat), pat)
        cls._syndrome_cache = table
        return table

    @classmethod
    def _g24_encode(cls, byte_val: int) -> int:
        """Return 24-bit Extended Golay codeword as an integer."""
        msg = byte_val << (cls._G23_MSG_BITS - cls.DATA_BITS)
        shifted = msg << cls._G23_PARITY_BITS
        rem = cls._poly_mod(shifted, cls._G23_GEN)
        code23 = shifted | rem
        parity = bin(code23).count('1') & 1
        return (code23 << 1) | parity

    @classmethod
    def _g24_decode(cls, code24: int) -> dict[str, Any]:
        code23 = code24 >> 1
        recv_parity = code24 & 1
        syn = cls._syndrome(code23)
        table = cls._syndrome_table()
        err_pat = table.get(syn)
        if err_pat is None:
            return {"valid": False, "uncorrectable": True, "byte": None,
                    "errors": 0, "syndrome": syn}
        corrected23 = code23 ^ err_pat
        err_positions = [p for p in range(cls._G23_CODE_BITS)
                         if err_pat & (1 << (cls._G23_CODE_BITS - 1 - p))]
        if (bin(corrected23).count('1') ^ recv_parity) & 1:
            err_positions.append(cls._G23_CODE_BITS)
        if len(err_positions) > cls._G23_MAX_CORRECT:
            return {"valid": False, "uncorrectable": True, "byte": None,
                    "errors": len(err_positions), "syndrome": syn}
        msg = corrected23 >> cls._G23_PARITY_BITS
        byte_val = msg >> (cls._G23_MSG_BITS - cls.DATA_BITS)
        return {"valid": True, "uncorrectable": False, "byte": byte_val,
                "errors": len(err_positions), "syndrome": syn}

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------

    @classmethod
    def encode(cls, byte_val: int) -> bytes:
        """Encode one byte → 6-byte (48-bit) codeword."""
        assert 0 <= byte_val <= 255
        code24 = cls._g24_encode(byte_val)
        # Pack 24-bit codeword into 3 bytes, repeat twice
        b0 = (code24 >> 16) & 0xFF
        b1 = (code24 >> 8) & 0xFF
        b2 = code24 & 0xFF
        return bytes([b0, b1, b2, b0, b1, b2])

    @classmethod
    def decode(cls, codeword: bytes) -> DecodeResult:
        """Decode 6-byte (48-bit) codeword → DecodeResult."""
        if len(codeword) != 6:
            raise ValueError(f"Expected 6 bytes, got {len(codeword)}")
        half_a = (codeword[0] << 16) | (codeword[1] << 8) | codeword[2]
        half_b = (codeword[3] << 16) | (codeword[4] << 8) | codeword[5]
        res_a = cls._g24_decode(half_a)
        res_b = cls._g24_decode(half_b)

        valid_a = res_a["valid"]
        valid_b = res_b["valid"]
        err_a = res_a["errors"]
        err_b = res_b["errors"]

        if valid_a and valid_b:
            byte_a, byte_b = res_a["byte"], res_b["byte"]
            total_err = err_a + err_b
            if byte_a == byte_b:
                conf = max(0.0, 1.0 - total_err * 0.1)
                return DecodeResult(
                    valid=True, payload=byte_a, confidence=conf,
                    errors_corrected=total_err, method=cls.METHOD,
                    uncorrectable=False,
                    detail={"half_a": res_a, "half_b": res_b, "agreement": True})
            # Disagree: trust the half with fewer corrections
            winner = res_a if err_a <= err_b else res_b
            return DecodeResult(
                valid=True, payload=winner["byte"], confidence=0.20,
                errors_corrected=min(err_a, err_b), method=cls.METHOD,
                uncorrectable=False,
                detail={"half_a": res_a, "half_b": res_b, "agreement": False})

        if valid_a:
            conf = max(0.0, 0.75 - err_a * 0.1)
            return DecodeResult(
                valid=True, payload=res_a["byte"], confidence=conf,
                errors_corrected=err_a, method=cls.METHOD, uncorrectable=False,
                detail={"half_a": res_a, "half_b": res_b, "agreement": None})

        if valid_b:
            conf = max(0.0, 0.75 - err_b * 0.1)
            return DecodeResult(
                valid=True, payload=res_b["byte"], confidence=conf,
                errors_corrected=err_b, method=cls.METHOD, uncorrectable=False,
                detail={"half_a": res_a, "half_b": res_b, "agreement": None})

        return DecodeResult(
            valid=False, payload=None, confidence=0.0, errors_corrected=0,
            method=cls.METHOD, uncorrectable=True,
            detail={"half_a": res_a, "half_b": res_b, "agreement": None})


# ===========================================================================
# Codec 2: RS48  —  Reed-Solomon RS(6,1) over GF(2^8)
# ===========================================================================

class RS48Codec:
    """
    Reed-Solomon RS(6,1) codec over GF(2^8).

    The 8-bit payload is treated as one GF(2^8) symbol.  The codeword
    comprises 6 symbols (48 bits):  [message, p4, p3, p2, p1, p0].
    Up to 2 symbol-errors (full byte corruptions) can be corrected,
    which may correspond to up to 16 bit-errors when concentrated in
    two separate RGB channels.

    Generator polynomial:  g(x) = (x+α)(x+α²)(x+α³)(x+α⁴)(x+α⁵)
    where α = 0x02 is the primitive element of GF(2^8) with polynomial 0x11D.

    Decode confidence
    -----------------
    - 0 symbol errors : 1.00
    - 1 symbol error  : 0.75
    - 2 symbol errors : 0.50
    - uncorrectable   : 0.00
    """

    METHOD = "rs48"
    CODEWORD_BITS = 48
    PIXELS_PER_BYTE = 2
    N_SYMBOLS = 6
    K_SYMBOLS = 1
    T_SYMBOLS = 2   # corrects up to 2 symbol (byte) errors

    # Generator polynomial coefficients [g5=1, g4, g3, g2, g1, g0]
    # Computed as product (x+α)(x+α²)(x+α³)(x+α⁴)(x+α⁵) over GF(2^8)
    _GEN_POLY: list[int] = []

    @classmethod
    def _build_gen_poly(cls) -> None:
        if cls._GEN_POLY:
            return
        _GF256._ensure_tables()
        roots = [_GF256.pow(2, i) for i in range(1, cls.N_SYMBOLS)]  # α¹..α⁵
        poly = [1]
        for root in roots:
            # Multiply poly by the factor (x + root) = [root, 1] in low-to-high form:
            #   coeff * x^i * root → position i
            #   coeff * x^i * x   → position i+1
            new_poly = [0] * (len(poly) + 1)
            for i, coeff in enumerate(poly):
                new_poly[i] ^= _GF256.mul(coeff, root)
                new_poly[i + 1] ^= coeff
            poly = new_poly
        cls._GEN_POLY = poly  # len = 6, index = degree; poly[5] = 1 (monic)

    @classmethod
    def _syndromes(cls, received: list[int]) -> list[int]:
        """Compute syndromes S_1 .. S_{2t} = S_1..S_4 (actually S_1..S_5 for t=2)."""
        _GF256._ensure_tables()
        syns = []
        for i in range(1, 2 * cls.T_SYMBOLS + 1 + 1):  # S_1..S_5
            alpha_i = _GF256.pow(2, i)
            s = 0
            for coeff in received:
                s = _GF256.mul(s, alpha_i) ^ coeff
            syns.append(s)
        return syns

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------

    @classmethod
    def encode(cls, byte_val: int) -> bytes:
        """Encode one byte → 6-byte (48-bit) RS codeword."""
        assert 0 <= byte_val <= 255
        cls._build_gen_poly()
        # Non-systematic codeword = m × g(x) in descending degree order:
        #   codeword = [m*g₅, m*g₄, m*g₃, m*g₂, m*g₁, m*g₀]
        #   where g₅ = 1 (monic), so first element is just m.
        codeword = [byte_val]
        for i in range(cls.N_SYMBOLS - 2, -1, -1):  # g₄ down to g₀
            codeword.append(_GF256.mul(byte_val, cls._GEN_POLY[i]))
        return bytes(codeword)

    @classmethod
    def decode(cls, codeword: bytes) -> DecodeResult:
        """Decode 6-byte (48-bit) RS codeword → DecodeResult."""
        if len(codeword) != cls.N_SYMBOLS:
            raise ValueError(f"Expected {cls.N_SYMBOLS} bytes, got {len(codeword)}")
        cls._build_gen_poly()
        received = list(codeword)
        syns = cls._syndromes(received)

        if all(s == 0 for s in syns):
            return DecodeResult(
                valid=True, payload=received[0], confidence=1.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=False,
                detail={"syndromes": syns})

        # Use Berlekamp-Massey to find error locator polynomial
        locator = _berlekamp_massey(
            syns[:2 * cls.T_SYMBOLS],
            _GF256.mul,
            _GF256.inv)

        num_errors = len(locator) - 1
        if num_errors > cls.T_SYMBOLS:
            return DecodeResult(
                valid=False, payload=None, confidence=0.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=True,
                detail={"syndromes": syns, "locator_degree": num_errors})

        # Chien search: find roots of Λ(x) among {α^{-i} : i=0..5}
        error_positions: list[int] = []
        for pos in range(cls.N_SYMBOLS):
            # Evaluate Λ at α^{-pos} = α^{255-pos}
            alpha_inv_pos = _GF256.pow(2, (255 - pos) % 255) if pos != 0 else 1
            val = 0
            for k, coeff in enumerate(locator):
                val ^= _GF256.mul(coeff, _GF256.pow(alpha_inv_pos, k))
            if val == 0:
                error_positions.append(pos)

        if len(error_positions) != num_errors:
            return DecodeResult(
                valid=False, payload=None, confidence=0.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=True,
                detail={"syndromes": syns, "chien_mismatch": True})

        # Forney algorithm: compute error magnitudes
        corrected = list(received)
        for pos in error_positions:
            # Λ'(x) = formal derivative of Λ (even powers vanish in GF(2^8))
            loc_deriv = [locator[i] for i in range(1, len(locator), 2)]
            alpha_pos = _GF256.pow(2, pos)
            alpha_inv_pos = _GF256.pow(2, (255 - pos) % 255) if pos != 0 else 1

            # Omega(x) = S(x)*Λ(x) mod x^{2t}
            s_poly = syns[:2 * cls.T_SYMBOLS]
            omega = [0] * (2 * cls.T_SYMBOLS)
            for i, lc in enumerate(locator):
                for j, sc in enumerate(s_poly):
                    if i + j < 2 * cls.T_SYMBOLS:
                        omega[i + j] ^= _GF256.mul(lc, sc)

            # Evaluate Omega at α^{-pos}
            omega_val = 0
            for k, coeff in enumerate(omega):
                omega_val ^= _GF256.mul(coeff, _GF256.pow(alpha_inv_pos, k))

            # Evaluate Λ' at α^{-pos}
            deriv_val = 0
            for k, coeff in enumerate(loc_deriv):
                deriv_val ^= _GF256.mul(coeff, _GF256.pow(alpha_inv_pos, k))

            if deriv_val == 0:
                return DecodeResult(
                    valid=False, payload=None, confidence=0.0,
                    errors_corrected=0, method=cls.METHOD, uncorrectable=True,
                    detail={"syndromes": syns, "forney_zero_derivative": pos})

            error_val = _GF256.mul(omega_val, _GF256.inv(deriv_val))
            # pos = polynomial degree of the error;
            # byte index in received[] is (N-1-pos)
            byte_idx = cls.N_SYMBOLS - 1 - pos
            corrected[byte_idx] ^= error_val

        conf_map = {0: 1.0, 1: 0.75, 2: 0.50}
        conf = conf_map.get(len(error_positions), 0.0)
        return DecodeResult(
            valid=True, payload=corrected[0], confidence=conf,
            errors_corrected=len(error_positions), method=cls.METHOD,
            uncorrectable=False,
            detail={"syndromes": syns, "error_positions": error_positions,
                    "corrected_symbols": corrected})


# ===========================================================================
# Codec 3: BCH48  —  Shortened BCH[63,24,15] + appended parity bit
# ===========================================================================

class BCH48Codec:
    """
    Shortened BCH[63,24,15] + 1 appended parity bit → [48, 8] code.

    Construction
    ------------
    * Full BCH[63, 24, 15] over GF(2^6):
      - primitive polynomial  p(x) = x^6 + x + 1  (0x43)
      - generator g(x) = m₁·m₃·m₅·m₇·m₉·m₁₁·m₁₃  (degree 39)
      - designed distance d = 15, corrects t = 7 errors
    * Shortening by 16 message bits → [47, 8, ≥15]
      - 8-bit message is padded with 16 leading zeros before encoding
      - first 16 bits of the 63-bit codeword are dropped
    * Extend with 1 overall parity bit → [48, 8, 16]
      - enables distinguishing odd/even error counts for improved detection

    Decode confidence
    -----------------
    - 0 errors       : 1.00
    - e errors ≤ 7   : 1.00 − e / 14
    - uncorrectable  : 0.00
    """

    METHOD = "bch48"
    CODEWORD_BITS = 48
    PIXELS_PER_BYTE = 2
    DATA_BITS = 8

    # BCH[63, 24, 15] parameters
    _N_FULL = 63
    _K_FULL = 24
    _T = 7           # error correction capability
    _N_SHORT = 47    # shortened n (= 63 - 16)
    _K_SHORT = 8     # shortened k (= 24 - 16)
    _N_EXT = 48      # after appending parity bit

    # Odd design roots: α¹, α³, α⁵, α⁷, α⁹, α¹¹, α¹³
    _DESIGN_ROOTS = [1, 3, 5, 7, 9, 11, 13]

    _generator_poly: int | None = None   # bit-vector (bit i = coeff of x^i)
    _gen_degree: int = 0

    # ------------------------------------------------------------------
    # GF(2^6) polynomial helpers  (coefficients are GF(2^6) elements)
    # ------------------------------------------------------------------

    @staticmethod
    def _poly_add_gf2(a: int, b: int) -> int:
        """Add two GF(2)[x] polynomials (= XOR of bit vectors)."""
        return a ^ b

    @staticmethod
    def _poly_mod_gf2(a: int, g: int) -> int:
        """Compute a mod g in GF(2)[x]."""
        g_deg = g.bit_length() - 1
        while True:
            a_deg = a.bit_length() - 1
            if a_deg < g_deg:
                break
            a ^= g << (a_deg - g_deg)
        return a

    @classmethod
    def _minimal_poly(cls, root_exp: int) -> int:
        """
        Compute the GF(2) minimal polynomial of α^{root_exp} as a bit-vector.
        """
        _GF64._ensure_tables()
        # Collect Frobenius orbit
        orbit: list[int] = []
        seen: set[int] = set()
        r = root_exp % _GF64.ORDER
        while r not in seen:
            seen.add(r)
            orbit.append(r)
            r = (2 * r) % _GF64.ORDER

        # Multiply (x + α^r) for r in orbit over GF(2^6)[x],
        # coefficients as list[GF(2^6)], index = degree
        poly_gf6: list[int] = [1]
        for r in orbit:
            root_val = _GF64.alpha_pow(r)
            new_poly = [0] * (len(poly_gf6) + 1)
            for i, coeff in enumerate(poly_gf6):
                new_poly[i + 1] ^= coeff
                new_poly[i] ^= _GF64.mul(coeff, root_val)
            poly_gf6 = new_poly

        # Coefficients must now be in GF(2) = {0, 1}
        result = 0
        for i, c in enumerate(poly_gf6):
            assert c in (0, 1), f"Non-GF(2) coefficient {c} at degree {i}"
            if c:
                result |= (1 << i)
        return result

    @classmethod
    def _build_generator(cls) -> None:
        if cls._generator_poly is not None:
            return
        g = 1  # = 1 (degree-0 polynomial in GF(2)[x])
        for root_exp in cls._DESIGN_ROOTS:
            m = cls._minimal_poly(root_exp)
            # Multiply g by m  (polynomial mult over GF(2))
            product = 0
            m_deg = m.bit_length() - 1
            g_deg = g.bit_length() - 1
            for i in range(g_deg + 1):
                if (g >> i) & 1:
                    product ^= m << i
            g = product
        cls._generator_poly = g
        cls._gen_degree = g.bit_length() - 1  # should be 39

    @classmethod
    def _encode_full63(cls, message_int: int) -> int:
        """
        Encode a 24-bit message integer into a BCH[63,24,15] codeword (63 bits).
        Systematic form: codeword = message * x^39 + (message * x^39 mod g).
        """
        cls._build_generator()
        shifted = message_int << cls._gen_degree
        parity = cls._poly_mod_gf2(shifted, cls._generator_poly)
        return shifted | parity

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------

    @classmethod
    def encode(cls, byte_val: int) -> bytes:
        """Encode one byte → 6-byte (48-bit) shortened BCH + parity codeword."""
        assert 0 <= byte_val <= 255
        # Pad 8-bit message with 16 leading zeros → 24-bit message
        full_codeword = cls._encode_full63(byte_val)  # 63-bit integer

        # Drop leading 16 bits → keep lower 47 bits (positions 0..46)
        # Full codeword: [23-bit message | 39-bit parity] as bit indices
        # After shortening: take bits 0..46 of the 63-bit codeword
        short_codeword = full_codeword & ((1 << cls._N_SHORT) - 1)

        # Append overall parity bit
        parity = bin(short_codeword).count('1') & 1
        ext_codeword = (short_codeword << 1) | parity  # 48 bits

        # Pack into 6 bytes big-endian
        return ext_codeword.to_bytes(6, 'big')

    @classmethod
    def _compute_syndromes(cls, received_47: int) -> list[int]:
        """
        Compute syndromes S_j = r(α^j) in GF(2^6) for j in design roots.
        received_47 is the 47-bit codeword (parity bit stripped).
        """
        _GF64._ensure_tables()
        syns: list[int] = []
        for j in range(1, 2 * cls._T + 1):  # S_1 .. S_14
            # Evaluate polynomial at α^j
            alpha_j = _GF64.alpha_pow(j)
            s = 0
            # Treat received_47 as polynomial; bit 46 = degree 46 coeff
            for deg in range(cls._N_SHORT - 1, -1, -1):
                s = _GF64.mul(s, alpha_j)
                if (received_47 >> deg) & 1:
                    s ^= 1
            syns.append(s)
        # S_{2j} = S_j^2 in binary BCH (Frobenius), verify consistency
        return syns

    @classmethod
    def decode(cls, codeword: bytes) -> DecodeResult:
        """Decode 6-byte (48-bit) shortened BCH codeword → DecodeResult."""
        if len(codeword) != 6:
            raise ValueError(f"Expected 6 bytes, got {len(codeword)}")
        cls._build_generator()

        ext_int = int.from_bytes(codeword, 'big')  # 48-bit integer
        received_parity = ext_int & 1
        received_47 = ext_int >> 1  # strip parity bit

        # Check overall parity
        actual_parity = bin(received_47).count('1') & 1
        parity_ok = (actual_parity == received_parity)

        # Compute syndromes for the 47-bit word
        syns = cls._compute_syndromes(received_47)
        all_zero = all(s == 0 for s in syns)

        if all_zero and parity_ok:
            # No errors
            payload = (received_47 >> cls._gen_degree) & 0xFF
            return DecodeResult(
                valid=True, payload=payload, confidence=1.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=False,
                detail={"syndromes": syns, "parity_ok": True})

        # BM algorithm over GF(2^6) using all 14 syndromes
        # For binary BCH we only pass odd syndromes + derive even via S_{2j}=S_j^2
        # But BM works on the full syndrome sequence S_1..S_{2t}
        locator = _berlekamp_massey(
            syns[:2 * cls._T],
            _GF64.mul,
            _GF64.inv)

        num_errors = len(locator) - 1
        if num_errors > cls._T:
            return DecodeResult(
                valid=False, payload=None, confidence=0.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=True,
                detail={"syndromes": syns, "locator_degree": num_errors,
                        "parity_ok": parity_ok})

        # Chien search: evaluate Λ(α^{-i}) = Λ(α^{63-i}) for i = 0..46
        _GF64._ensure_tables()
        error_positions: list[int] = []
        for pos in range(cls._N_SHORT):
            # α^{-pos} = α^{63-pos mod 63}
            exp = (cls._N_FULL - pos) % cls._N_FULL
            alpha_inv_pos = _GF64.alpha_pow(exp) if pos != 0 else 1
            val = 0
            for k, coeff in enumerate(locator):
                val ^= _GF64.mul(coeff, _GF64.pow(alpha_inv_pos, k))
            if val == 0:
                error_positions.append(pos)

        if len(error_positions) != num_errors:
            return DecodeResult(
                valid=False, payload=None, confidence=0.0,
                errors_corrected=0, method=cls.METHOD, uncorrectable=True,
                detail={"syndromes": syns, "chien_mismatch": True,
                        "found_positions": error_positions})

        # Flip error bits in received_47
        corrected_47 = received_47
        for pos in error_positions:
            corrected_47 ^= (1 << pos)

        # Extract 8-bit payload: message bits are at positions [39..46]
        payload = (corrected_47 >> cls._gen_degree) & 0xFF

        conf = max(0.0, 1.0 - num_errors / 14.0)
        return DecodeResult(
            valid=True, payload=payload, confidence=conf,
            errors_corrected=num_errors, method=cls.METHOD, uncorrectable=False,
            detail={"syndromes": syns, "error_positions": error_positions,
                    "parity_ok": parity_ok})


# ===========================================================================
# Factory
# ===========================================================================

_CODEC_MAP: dict[str, type] = {
    "golay24-dual": Golay24DualCodec,
    "rs48": RS48Codec,
    "bch48": BCH48Codec,
}

SUPPORTED_METHODS: tuple[str, ...] = tuple(_CODEC_MAP.keys())


def get_codec(method: str) -> type:
    """
    Return the codec class for the given method name.

    Parameters
    ----------
    method : str
        One of ``'golay24-dual'``, ``'rs48'``, ``'bch48'``.

    Returns
    -------
    type
        The codec class (``Golay24DualCodec``, ``RS48Codec``, or ``BCH48Codec``).
    """
    if method not in _CODEC_MAP:
        raise ValueError(
            f"Unknown ECC48 method '{method}'. "
            f"Supported: {list(_CODEC_MAP.keys())}")
    return _CODEC_MAP[method]
