import matplotlib.pyplot as plt 
packages = ['streamlit', 'scikit-learn', 'numpy', 'joblib', 'pandas'] 
weights = [40, 25, 20, 10, 5] 
plt.figure(figsize=(8, 6)) 
plt.bar(packages, weights, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB3E6']) 
plt.xlabel('Packages') 
plt.ylabel('Weightage (%)') 
plt.title('Dependency Weightage for Diabetic Retinopathy App') 
plt.ylim(0, 50) 
for i, v in enumerate(weights): 
    plt.text(i, v + 1, f"{v}%%", ha='center', fontweight='bold') 
plt.tight_layout() 
plt.savefig('dependency_weightage.png', dpi=300) 
plt.close() 
