#!/usr/bin/env python3
# nist_pwgen_utf8.py
import argparse, secrets, sys, pathlib, unicodedata, string

PRINTABLE_ASCII = ''.join(chr(c) for c in range(32, 127))  # space(0x20)~'~'(0x7E)

def normalize(s: str, form: str) -> str:
    return s if form.lower() == "none" else unicodedata.normalize(form.upper(), s)

def is_nfkc_stable_char(ch: str, form="NFKC") -> bool:
    # 정규화 후 자기 자신이고, 제어문자/결합표식이 아닌지 체크
    if normalize(ch, form) != ch:
        return False
    cat = unicodedata.category(ch)
    if cat.startswith('C'):  # Cc, Cf, Cs, Co, Cn (제어·비지정 등) 제거
        return False
    if cat.startswith('M'):  # Mn, Mc, Me (결합표식) 제거
        return False
    return True

def build_unicode_safe_alphabet() -> str:
    # 과도하게 넓은 유니코드 전 범위 대신, 대표 스크립트들에서 NFKC-stable 문자만 수집
    ranges = [
        (0x0020, 0x007E),  # Basic Latin (ASCII printable)
        (0x00A1, 0x024F),  # Latin-1/Extended
        (0x0370, 0x03FF),  # Greek
        (0x0400, 0x04FF),  # Cyrillic
        (0x3041, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0xAC00, 0xD7A3),  # Hangul Syllables
    ]
    out = []
    for a, b in ranges:
        for cp in range(a, b + 1):
            ch = chr(cp)
            if is_nfkc_stable_char(ch, "NFKC"):
                out.append(ch)
    # 가독성 떨어지는 따옴표/역슬래시 등은 제외(원하면 아래 주석 해제)
    # ambiguous = set('\'"`\\')
    # out = [c for c in out if c not in ambiguous]
    return ''.join(out)

def load_wordlist(path: pathlib.Path, norm: str):
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        words = [normalize(w.strip(), norm) for w in f if w.strip() and not w.startswith('#')]
    return words

def load_blocklist(path: pathlib.Path, norm: str):
    if not path:
        return set()
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return set(normalize(line.strip(), norm) for line in f if line.strip())

def gen_random(length, alphabet):
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def gen_passphrase(num_words, sep, wordlist):
    return sep.join(secrets.choice(wordlist) for _ in range(num_words))

def main():
    p = argparse.ArgumentParser(description="NIST 800-63B(-4) UTF-8/Unicode password generator")
    p.add_argument("--context", choices=["single","mfa"], default="single",
                   help="single: 비밀번호 단독(기본 15자), mfa: MFA 일부(기본 8자)")
    p.add_argument("--mode", choices=["random","passphrase"], default="passphrase")
    p.add_argument("--length", type=int, default=None,
                   help="mode=random일 때 길이(문자 수). 기본: single=20, mfa=12")
    p.add_argument("--alphabet", choices=["ascii","unicode_safe"], default="unicode_safe",
                   help="unicode_safe: 다국어 안전 문자집합(NFKC-stable), ascii: 인쇄가능 ASCII")
    p.add_argument("--wordlist", type=pathlib.Path, help="패스프레이즈용 단어 목록(UTF-8, 한 줄 한 단어)")
    p.add_argument("--words", type=int, default=6, help="패스프레이즈 단어 개수")
    p.add_argument("--sep", default="-", help="패스프레이즈 구분자")
    p.add_argument("--max-length", type=int, default=64, help="허용 최대 길이(문자 수)")
    p.add_argument("--count", type=int, default=3, help="몇 개 생성")
    p.add_argument("--blocklist", type=pathlib.Path, help="블록리스트 파일(UTF-8, 한 줄 한 값)")
    p.add_argument("--norm", choices=["NFKC","NFKD","NFC","none"], default="NFKC",
                   help="입력/대조/검사에 적용할 유니코드 정규화(기본 NFKC)")
    p.add_argument("--show-utf8", action="store_true",
                   help="생성값의 UTF-8 바이트(hex) 출력(검증·해시 파이프라인 점검용)")
    args = p.parse_args()

    # NIST Rev.4 운영 가정: 단독 15자 / MFA 8자 (필요 시 정책에 맞게 변경)
    min_len = 15 if args.context == "single" else 8

    # 알파벳 구성
    if args.mode == "random":
        if args.alphabet == "ascii":
            alphabet = PRINTABLE_ASCII
        else:
            alphabet = build_unicode_safe_alphabet()
        default_len = 20 if args.context == "single" else 12
        length = args.length or default_len
        # 길이 검사는 "문자 수(코드포인트 수)" 기준 (NIST 요구)
        if length < min_len:
            print(f"[err] length {length} < minimum {min_len}", file=sys.stderr); sys.exit(2)
        if length > args.max_length:
            print(f"[err] length {length} > max-length {args.max_length}", file=sys.stderr); sys.exit(2)

        block = load_blocklist(args.blocklist, args.norm)
        for _ in range(args.count):
            while True:
                candidate = gen_random(length, alphabet)
                canon = normalize(candidate, args.norm)
                # 정규화 후 길이 재확인(정규화로 길이가 달라질 수 있음)
                if len(canon) < min_len or len(canon) > args.max_length:
                    continue
                if canon not in block:
                    break
            print(canon)
            # if args.show-utf8:
            #     print("# utf8:", canon.encode("utf-8").hex(), file=sys.stderr)

    else:  # passphrase
        if not args.wordlist or not args.wordlist.exists():
            print("[err] --wordlist 를 제공하세요 (UTF-8, 한 줄 한 단어).", file=sys.stderr); sys.exit(2)
        words = load_wordlist(args.wordlist, args.norm)
        block = load_blocklist(args.blocklist, args.norm)

        for _ in range(args.count):
            for _try in range(10000):
                phrase = gen_passphrase(args.words, args.sep, words)
                canon = normalize(phrase, args.norm)
                if len(canon) >= min_len and len(canon) <= args.max_length and canon not in block:
                    break
            else:
                print("[err] 적절한 길이로 생성 실패: --words/--sep/--max-length 조정.", file=sys.stderr); sys.exit(2)
            print(canon)
            # if args.show-utf8:
            #     print("# utf8:", canon.encode("utf-8").hex(), file=sys.stderr)

if __name__ == "__main__":
    main()