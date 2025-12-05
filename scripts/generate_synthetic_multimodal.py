import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_samples=2000):
    print(f"Generating {n_samples} synthetic patient records...")
    
    # 1. Generate Basic Features
    ids = [f"P{str(i).zfill(5)}" for i in range(n_samples)]
    
    # Age: Normal distribution around 45, min 18, max 90
    age = np.random.normal(45, 15, n_samples)
    age = np.clip(age, 18, 90).astype(int)
    
    # Gender: 0 (Male), 1 (Female)
    gender = np.random.randint(0, 2, n_samples)
    
    # BMI: Normal distribution around 25
    bmi = np.random.normal(25, 5, n_samples)
    bmi = np.clip(bmi, 15, 45)
    
    # Smoking: 30% smokers
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # 2. Generate Multimodal/Complex Features (Simulating Image/Text Embeddings)
    # CXR_Opacity: Higher means worse lungs (correlated with Smoking and Age)
    # Base randomness + effect of smoking + effect of age
    cxr_opacity = np.random.normal(0.3, 0.1, n_samples) + (smoking * 0.2) + (age / 200)
    cxr_opacity = np.clip(cxr_opacity, 0, 1)
    
    # Genetic_Marker: Random genetic predisposition (0 to 1)
    genetic_marker = np.random.rand(n_samples)
    
    # 3. Define "Ground Truth" Logic (The Rules the Model Must Learn)
    
    # --- LOGIC FOR RISK (Classification) ---
    # Risk Probability = Sigmoid(Linear Combination of features)
    # Rule: Age, Smoking, and High CXR Opacity increase risk significantly.
    risk_score_logit = (
        -5.0 +                  # Base bias (makes default risk low)
        (age * 0.05) +          # Older = Higher Risk
        (bmi * 0.05) +          # Higher BMI = Higher Risk
        (smoking * 2.0) +       # Smoking = Huge Risk Factor
        (cxr_opacity * 3.0) +   # Bad Image Features = High Risk
        (genetic_marker * 1.0)  # Genetic factor
    )
    
    # Convert logit to probability (Sigmoid)
    risk_prob = 1 / (1 + np.exp(-risk_score_logit))
    
    # Assign Class based on probability (Threshold 0.5)
    # Adding a tiny bit of noise so it's not perfectly separable (Realistic)
    risk_class = [1 if p > 0.5 else 0 for p in risk_prob]
    
    # --- LOGIC FOR COST (Regression) ---
    # Rule: Cost is heavily driven by Opacity (Severity) and Age.
    # Base Cost: 1000 USD
    # Age adds: $100 per year
    # Smoking adds: $5000 (Complications)
    # CXR Opacity (Severity) adds: up to $20,000
    treatment_cost = (
        1000 + 
        (age * 100) + 
        (bmi * 200) + 
        (smoking * 5000) + 
        (cxr_opacity * 20000) +
        np.random.normal(0, 1000, n_samples) # Add random noise (uncertainty)
    )
    
    # 4. Create DataFrame
    df = pd.DataFrame({
        'Patient_ID': ids,
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Smoking': smoking,
        'Genetic_Marker': genetic_marker,
        'CXR_Opacity': cxr_opacity, # Representing an image feature
        'Risk_Class': risk_class,   # Target 1
        'Treatment_Cost': treatment_cost # Target 2
    })
    
    return df

if __name__ == "__main__":
    # Generate
    df = generate_data(2000)
    
    # Save
    output_path = 'data/multimodal.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated 2000 patient records.")
    print(f"Data saved to: {output_path}")
    print("\n--- Data Sample ---")
    print(df.head())
    print("\n--- Logic Summary (Ground Truth) ---")
    print("Risk Logic: Driven by Age, Smoking, and Lung Opacity.")
    print("Cost Logic: Base + (Age*100) + (Smoking*5000) + (Opacity*20000).")
