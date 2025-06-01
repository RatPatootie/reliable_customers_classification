Business needs

    This project provides a machine learning model that predicts the likelihood of a customer defaulting on a loan, allowing the bank to make informed decisions and mitigate financial risks.
    
Requirements

    python 3.12

Dependencies:

    numpy==1.26.4
    pandas==2.2.2
    sklearn==1.5.2
    gdown==5.2.0

To install the required dependencies, run:

    pip install -r requirements.txt
    
Preparation:

    before running predict.py you should run devide_data.py for dowloading and  dividing data on train and new_input

Running:

    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'Status' column with the result value.

    The input is expected  csv file in the 'data' folder with a name <new_input.csv>. The file should have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train.csv> should contain all features columns and target for prediction Status.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" and "scaler.plk" will be created.

    Run the training script:
        python train.py

    
    The model accuracy is 72%