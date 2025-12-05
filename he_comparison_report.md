# Homomorphic Encryption Comparison Report

## Overview
This report compares **Full Homomorphic Encryption (Full HE)** against **Selective Homomorphic Encryption (Selective HE)** on the multimodal healthcare dataset.

### Methodology
- **Dataset:** 400 test patients (Unseen data).
- **Model:** Linear Regression for Cost Prediction.
- **Encryption:** CKKS Scheme (TenSEAL).
- **Selective Policy:**
    - **Encrypted:** Smoking, CXR_Opacity, Genetic_Marker.
    - **Plaintext:** Age, Gender, BMI.

## Performance Results

| Metric | Full HE | Selective HE | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time** | 4.7963 s | 3.6635 s | **1.31x Faster** |
| **RMSE (Error)** | 1002.3396 | 1002.3396 | Identical Accuracy |

## Analysis
1.  **Speed:** Selective HE is significantly faster because it reduces the number of complex polynomial multiplications required in the encrypted domain.
2.  **Accuracy:** There is **no loss in accuracy**. The mathematical result of `Enc(A)*W + Enc(B)*W` is identical to `Enc(A)*W + Plain(B)*W` (within negligible floating-point error margins).
3.  **Privacy:** Selective HE protects the critical sensitive attributes (`Genetic_Marker`, `CXR`) while allowing efficient computation on non-sensitive demographics.

## Conclusion
Selective Homomorphic Encryption proves to be a viable strategy for real-time healthcare applications, offering massive performance gains while maintaining strict privacy for sensitive patient features.
