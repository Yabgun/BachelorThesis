import hashlib

VECTORS = [
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
]


def main():
    ok = True
    for msg, expected_hex in VECTORS:
        got_hex = hashlib.sha256(msg).hexdigest()
        if got_hex.lower() == expected_hex.lower():
            print("SHA-256 vector PASS:", msg[:20])
        else:
            print("SHA-256 vector FAIL:", msg[:20])
            print("expected:", expected_hex)
            print("got     :", got_hex)
            ok = False
    if ok:
        print("All SHA-256 vectors: PASS")
    else:
        print("Some SHA-256 vectors failed")


if __name__ == '__main__':
    main()