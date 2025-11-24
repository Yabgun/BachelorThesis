from Crypto.Cipher import AES

# NIST SP 800-38A F.1.1 AES-128-ECB test vector
KEY_HEX = "2b7e151628aed2a6abf7158809cf4f3c"
PT_HEX  = (
    "6bc1bee22e409f96e93d7e117393172a"
    "ae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52ef"
    "f69f2445df4f9b17ad2b417be66c3710"
)
CT_HEX  = (
    "3ad77bb40d7a3660a89ecaf32466ef97"
    "f5d3d58503b9699de785895a96fdbaaf"
    "43b1cd7f598ece23881b00e3ed030688"
    "7b0c785e27e8ad3f8223207104725dd4"
)


def main():
    key = bytes.fromhex(KEY_HEX)
    pt = bytes.fromhex(PT_HEX)
    expected_ct = CT_HEX

    cipher = AES.new(key, AES.MODE_ECB)
    ct = cipher.encrypt(pt)
    got = ct.hex()

    if got.lower() == expected_ct.lower():
        print("AES-128-ECB NIST vector: PASS")
    else:
        print("AES-128-ECB NIST vector: FAIL")
        print("expected:", expected_ct)
        print("got     :", got)


if __name__ == '__main__':
    main()