from Crypto.Cipher import AES

# NIST SP 800-38A F.5.1 AES-128-CTR test vector
KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
CTR_HEX = "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
PT_HEX  = (
    "6bc1bee22e409f96e93d7e117393172a"
    "ae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52ef"
    "f69f2445df4f9b17ad2b417be66c3710"
)
CT_HEX  = (
    "874d6191b620e3261bef6864990db6ce"
    "9806f66b7970fdff8617187bb9fffdff"
    "5ae4df3edbd5d35e5b4f09020db03eab"
    "1e031dda2fbe03d1792170a0f3009cee"
)


def main():
    key = bytes.fromhex(KEY_HEX)
    ctr = bytes.fromhex(CTR_HEX)
    pt = bytes.fromhex(PT_HEX)

    expected_ct = CT_HEX

    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=ctr)
    ct = cipher.encrypt(pt)
    got = ct.hex()

    if got.lower() == expected_ct.lower():
        print("AES-128-CTR NIST vector: PASS")
    else:
        print("AES-128-CTR NIST vector: FAIL")
        print("expected:", expected_ct)
        print("got     :", got)


if __name__ == '__main__':
    main()