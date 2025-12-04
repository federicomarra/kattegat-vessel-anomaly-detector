# Main HPC Script to run all steps sequentially

# Run this script to install required packages
# pip install -r requirements.txt

# File imports
import main_1_data as data
import main_2_pre_processing as pre_proc
#import main_3_training as train
#import main_4_testing as test

import training_run as train_run
import testing_run as test_run

if __name__ == "__main__":
    # Decide which days to run in config.py
    
    # Run all steps sequentially
    data.main_data()
    pre_proc.main_pre_processing()
    #train.main_training()
    #test.main_testing()
    train_run.training_run()
    test_run.testing_run()