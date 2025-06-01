mean_impute_columns = [
                     'property_value',
                     'income',
                     ]

mode_impute_columns = ['loan_limit',
                     'loan_purpose',
                     'term',
                     'Neg_ammortization',
                     'age',
                     'submission_of_application',
                     ]

outlier_columns = [
                   'loan_amount','income',
                   'property_value']

cat_columns = ['age',
             'loan_limit',
             'Credit_Worthiness',
             'business_or_commercial',
             'Neg_ammortization',
             'interest_only',
             'lump_sum_payment',
             'occupancy_type',
             'submission_of_application',
             'co-applicant_credit_type',
             'loan_purpose',
             'loan_type',
             ]

X_columns = ['loan_limit', 
       'Credit_Worthiness', 'business_or_commercial',
       'loan_amount','property_value',
       'term', 'Neg_ammortization', 'interest_only',
       'lump_sum_payment','occupancy_type',
       'income',
       'Credit_Score', 'submission_of_application',
       'Gender_Male',
       'loan_type', 'loan_purpose', 
       'credit_type_EXP', 'co-applicant_credit_type',
       'age',]

hot_columns=[
              'Gender',
              'credit_type',
]
normis = [
    'loan_amount', 
    'term', 
    'property_value', 
    'occupancy_type', 
    'income', 
    'Credit_Score', 
    'age'
]
y_column = ['Status'] # target variable

