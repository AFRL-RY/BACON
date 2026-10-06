import numpy as np
import pandas as pd

def get_test_accuracy(test_file):

    test_data_df = pd.read_csv(test_file)
    test_file_acc = (test_data_df['Label'] == test_data_df['Predicted']).astype(int).mean()
    
    return test_file_acc