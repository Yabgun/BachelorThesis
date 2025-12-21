import tenseal as ts
import numpy as np

def demonstrate_randomness():
    # 1. Context (Ortam) Oluşturma
    # Bu ayarlar (8192, [60, 40...]) SABİTTİR. Kurallardır.
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    print("\n--- DENEY BAŞLIYOR ---")
    
    # Şifreleyeceğimiz Veri
    data = [0.75]
    print(f"Ham Veri: {data}")

    # 2. İlk Şifreleme
    enc_1 = ts.ckks_vector(context, data)
    # Şifreli verinin binary (makine kodu) halinin ilk 20 karakterini alalım
    bytes_1 = enc_1.serialize()[:20] 
    
    # 3. İkinci Şifreleme (AYNI VERİ, AYNI CONTEXT)
    enc_2 = ts.ckks_vector(context, data)
    bytes_2 = enc_2.serialize()[:20]

    print(f"\n1. Şifreleme İmzası (Hex): {bytes_1.hex()}")
    print(f"2. Şifreleme İmzası (Hex): {bytes_2.hex()}")

    # 4. Karşılaştırma
    if bytes_1 != bytes_2:
        print("\nSONUÇ: Gördüğün gibi, aynı veriyi şifrelesek bile şifreli halleri BAMBAŞKA!")
        print("Bu 'Rastgelelik' (Probabilistic Encryption) özelliğidir. Güvenlik için şarttır.")
    else:
        print("\nSONUÇ: Veriler aynı.")

    # 5. Doğrulama (Decryption)
    dec_1 = enc_1.decrypt()[0]
    dec_2 = enc_2.decrypt()[0]
    
    print(f"\nAncak şifreleri çözdüğümüzde:")
    print(f"1. Çözülen: {dec_1:.10f}")
    print(f"2. Çözülen: {dec_2:.10f}")
    print("Sonuçlar matematiksel olarak aynıdır (küçük float farkları hariç).")

if __name__ == "__main__":
    demonstrate_randomness()
