# pylint: disable=all
# md5_pure.py
# Python 3.6+  (no external deps)

import struct
import math

# 라운드별 순환 좌회전 수 (RFC 1321)
_S = [7,12,17,22]*4 + [5,9,14,20]*4 + [4,11,16,23]*4 + [6,10,15,21]*4

# K[i] = floor(2^32 * |sin(i+1)|)
_K = [int((1 << 32) * abs(math.sin(i + 1))) & 0xFFFFFFFF for i in range(64)]

def _rotl32(x, s):
    return ((x << s) | (x >> (32 - s))) & 0xFFFFFFFF

class MD5:
    """Minimal hashlib-like MD5 (educational)."""
    __slots__ = ('_A','_B','_C','_D','_count','_buffer')

    def __init__(self, data=b''):
        # 초기값(IV) - 리틀엔디안
        self._A = 0x67452301
        self._B = 0xEFCDAB89
        self._C = 0x98BADCFE
        self._D = 0x10325476
        self._count = 0        # 지금까지 받은 바이트 수
        self._buffer = b''     # 64바이트 미만 잔여
        if data:
            self.update(data)

    def update(self, data):
        if not data:
            return self
        self._count += len(data)
        data = self._buffer + data
        # 64바이트(512비트) 블록 단위 압축
        for off in range(0, (len(data) // 64) * 64, 64):
            self._compress(data[off:off+64])
        self._buffer = data[(len(data)//64)*64:]
        return self

    def _compress(self, block64):
        A, B, C, D = self._A, self._B, self._C, self._D
        M = list(struct.unpack('<16I', block64))  # 16개 32비트(LE)

        a, b, c, d = A, B, C, D
        for i in range(64):
            if i < 16:
                f = (b & c) | (~b & d)
                g = i
            elif i < 32:
                f = (d & b) | (~d & c)
                g = (5*i + 1) & 0xF
            elif i < 48:
                f = b ^ c ^ d
                g = (3*i + 5) & 0xF
            else:
                f = c ^ (b | ~d)
                g = (7*i) & 0xF

            tmp = (a + f + _K[i] + M[g]) & 0xFFFFFFFF
            a, d, c, b = d, c, b, (b + _rotl32(tmp, _S[i])) & 0xFFFFFFFF

        # 피드포워드(모듈러 2^32)
        self._A = (self._A + a) & 0xFFFFFFFF
        self._B = (self._B + b) & 0xFFFFFFFF
        self._C = (self._C + c) & 0xFFFFFFFF
        self._D = (self._D + d) & 0xFFFFFFFF

    def _finalize(self):
        # 현재 상태 복사 후 패딩/길이 부착 처리(원 상태 보존)
        A, B, C, D = self._A, self._B, self._C, self._D
        buf = self._buffer
        r = len(buf)

        # 패딩: 0x80 + 0x00*pad_len  (r+1+pad_len) % 64 == 56
        pad_len = (56 - (r + 1)) % 64
        tail = buf + b'\x80' + (b'\x00' * pad_len)
        # 원본 길이(비트) 64비트 LE
        tail += struct.pack('<Q', (self._count * 8) & 0xFFFFFFFFFFFFFFFF)

        # 남은 블록 압축
        for off in range(0, len(tail), 64):
            M = list(struct.unpack('<16I', tail[off:off+64]))
            a, b, c, d = A, B, C, D
            for i in range(64):
                if i < 16:
                    f = (b & c) | (~b & d)
                    g = i
                elif i < 32:
                    f = (d & b) | (~d & c)
                    g = (5*i + 1) & 0xF
                elif i < 48:
                    f = b ^ c ^ d
                    g = (3*i + 5) & 0xF
                else:
                    f = c ^ (b | ~d)
                    g = (7*i) & 0xF
                tmp = (a + f + _K[i] + M[g]) & 0xFFFFFFFF
                a, d, c, b = d, c, b, (b + _rotl32(tmp, _S[i])) & 0xFFFFFFFF
            A = (A + a) & 0xFFFFFFFF
            B = (B + b) & 0xFFFFFFFF
            C = (C + c) & 0xFFFFFFFF
            D = (D + d) & 0xFFFFFFFF

        return struct.pack('<4I', A, B, C, D)

    def digest(self):
        return self._finalize()

    def hexdigest(self):
        return self._finalize().hex()

# 간단 헬퍼
def md5_hexdigest(data: bytes) -> str:
    return MD5(data).hexdigest()
