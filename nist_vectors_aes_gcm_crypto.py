from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# NIST SP 800-38D example with no AAD
KEY_HEX = "00000000000000000000000000000000"
IV_HEX  = "000000000000000000000000"
PT_HEX  = "00000000000000000000000000000000"
CT_HEX  = "0388dace60b6a392f328c2b971b2fe78"
TAG_HEX = "ab6e47d42cec13bdf53a67b21257bddf"


def main():
    key = bytes.fromhex(KEY_HEX)
    nonce = bytes.fromhex(IV_HEX)
    data = bytes.fromhex(PT_HEX)

    aead = AESGCM(key)
    ct_and_tag = aead.encrypt(nonce, data, None)
    ct = ct_and_tag[:-16]
    tag = ct_and_tag[-16:]

    got_ct = ct.hex()
    got_tag = tag.hex()

    ok_ct = got_ct.lower() == CT_HEX.lower()
    ok_tag = got_tag.lower() == TAG_HEX.lower()

    if ok_ct and ok_tag:
        print("AES-128-GCM (cryptography) NIST vector: PASS")
    else:
        print("AES-128-GCM (cryptography) NIST vector: FAIL")
        print("expected CT:", CT_HEX)
        print("got      CT:", got_ct)
        print("expected TAG:", TAG_HEX)
        print("got      TAG:", got_tag)


if __name__ == '__main__':
    main()