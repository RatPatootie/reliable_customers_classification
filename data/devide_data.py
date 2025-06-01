import pandas as pd
from sklearn.model_selection import train_test_split
import gdown
url ="https://drive.google.com/file/d/1iP6k5VuFn0pMEPioBuY1PFOVvd2TNMyX/view?usp=sharing"
gdown.download(url, "data/variant_1.csv", quiet=False, fuzzy=True)
ds = pd.read_csv("data/variant_1.csv")

X_train, X_test = train_test_split(ds, train_size=0.9)

# Select relevant columns for the new input file
X_columns = ['loan_limit', 
       'Credit_Worthiness', 'business_or_commercial',
       'loan_amount','property_value',
       'term', 'Neg_ammortization', 'interest_only',
       'lump_sum_payment','occupancy_type',
       'income',
       'Credit_Score', 'submission_of_application',
       'Gender',
       'loan_type', 'loan_purpose', 
       'credit_type', 'co-applicant_credit_type',
       'age',]
 
# Save the training and test datasets
X_train.to_csv('data/train.csv', index=False)
X_test[X_columns].to_csv('data/new_input.csv', index=False)