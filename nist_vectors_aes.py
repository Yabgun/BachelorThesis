from Crypto.Cipher import AES

# NIST SP 800-38A F.2.1 AES-128-CBC test vector
KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
IV_HEX  = "000102030405060708090a0b0c0d0e0f"
PT_HEX  = (
    "6bc1bee22e409f96e93d7e117393172a"
    "ae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52ef"
    "f69f2445df4f9b17ad2b417be66c3710"
)
CT_HEX  = (
    "7649abac8119b246cee98e9b12e9197d"
    "5086cb9b507219ee95db113a917678b2"
    "73bed6b8e3c1743b7116e69e22229516"
    "3ff1caa1681fac09120eca307586e1a7"
)


def main():
    key = bytes.fromhex(KEY_HEX)
    iv = bytes.fromhex(IV_HEX)
    pt = bytes.fromhex(PT_HEX)
    expected_ct = CT_HEX

    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    ct = cipher.encrypt(pt)
    got = ct.hex()

    if got.lower() == expected_ct.lower():
        print("AES-128-CBC NIST vector: PASS")
    else:
        print("AES-128-CBC NIST vector: FAIL")
        print("expected:", expected_ct)
        print("got     :", got)


if __name__ == '__main__':
    main()