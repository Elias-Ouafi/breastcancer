from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

# Save to CSV
df.to_csv("data.csv", index=False)
