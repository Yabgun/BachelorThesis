from Crypto.Cipher import AES

# Known AES-128-GCM test vector with AAD (all-zero 16 bytes)
# Source: NIST SP 800-38D sample
# Key = 00000000000000000000000000000000
# IV  = 000000000000000000000000
# AAD = 00000000000000000000000000000000
# PT  = 00000000000000000000000000000000
# CT  = 0388dace60b6a392f328c2b971b2fe78
# TAG = ab6e47d42cec13bdf53a67b21257bddf
KEY_HEX = "00000000000000000000000000000000"
IV_HEX  = "000000000000000000000000"
AAD_HEX = "00000000000000000000000000000000"
PT_HEX  = "00000000000000000000000000000000"
CT_HEX  = "0388dace60b6a392f328c2b971b2fe78"
TAG_HEX = "ab6e47d42cec13bdf53a67b21257bddf"


def main():
    key = bytes.fromhex(KEY_HEX)
    iv = bytes.fromhex(IV_HEX)
    aad = bytes.fromhex(AAD_HEX)
    pt = bytes.fromhex(PT_HEX)
    expected_ct = CT_HEX
    expected_tag = TAG_HEX

    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    cipher.update(aad)
    ct, tag = cipher.encrypt_and_digest(pt)
    got_ct_hex = ct.hex()
    got_tag_hex = tag.hex()

    ok_ct = got_ct_hex.lower() == expected_ct.lower()
    ok_tag = got_tag_hex.lower() == expected_tag.lower()

    if ok_ct and ok_tag:
        print("AES-128-GCM NIST vector: PASS")
    else:
        print("AES-128-GCM NIST vector: FAIL")
        print("expected CT:", expected_ct)
        print("got      CT:", got_ct_hex)
        print("expected TAG:", expected_tag)
        print("got      TAG:", got_tag_hex)


if __name__ == '__main__':
    main()