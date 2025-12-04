# Main test script

# File imports
import config as config_file
from src.test.ais_tester import AISTester

# Library imports
import os
import json


def main_test():
    """
    Orchestrates the testing and evaluation pipeline for a trained deep learning model.
    This function performs the following steps:
    1.  **Configuration Setup**: Defines model parameters, input data paths (Parquet files), and output directories based on a global configuration file.
    2.  **Model Loading**: Loads the model configuration (JSON) and pre-trained weights (.pth) to initialize the `AISTester` class.
    3.  **Data Loading & Evaluation**: Loads the test dataset and runs inference to calculate error metrics.
    4.  **Visualization & Reporting**:
        *   Plots error distributions (histograms/boxplots) for the entire dataset.
        *   Identifies and plots the best and worst performing segments based on reconstruction error.
        *   Generates geographic maps visualizing the trajectory reconstruction for best, worst, and random segments.
    Global Variables Used (via `config_file`):
        N_BEST_WORST (int): Number of best/worst examples to plot.
        N_MAP_RANDOM (int): Number of random examples to map.
        PRE_PROCESSING_DF_TEST_PATH (str): Path to the test dataset parquet file.
        TEST_OUTPUT_DIR (str): Base directory for saving test results.
        TRAIN_OUTPUT_DIR (str): Directory containing trained model weights and configs.
    """
    
    # --- 1. CONFIGURATION ---
    
    # Name of the model configuration to use
    MODEL_NAME = "H128_L16_Lay1_lr0.001_BS64_Drop0.0_mse"  # Change as needed
    
    N_BEST_WORST = config_file.N_BEST_WORST
    N_MAP_RANDOM = config_file.N_MAP_RANDOM
    
    # Data to test on
    PARQUET_FILE = config_file.PRE_PROCESSING_DF_TEST_PATH

    # Output Directory
    OUTPUT_DIR = config_file.TEST_OUTPUT_DIR + "/" + MODEL_NAME
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    WEIGHTS_FILE = config_file.TRAIN_OUTPUT_DIR + "/weights_" + MODEL_NAME + ".pth"
    MODEL_CONFIG_FILE = config_file.TRAIN_OUTPUT_DIR + "/config_" + MODEL_NAME + ".json"
    
    
    # --- 2. LOAD MODEL AND INIT TESTER ---
    
    # Load Model Config
    with open(MODEL_CONFIG_FILE, 'r') as f:
        model_config = json.load(f)

    # Initialize Tester
    tester = AISTester(model_config, WEIGHTS_FILE, output_dir=OUTPUT_DIR)
    
    
    # --- 3. RUN TESTING AND EVALUATION ---
    
    if os.path.exists(PARQUET_FILE):
        # 1. Evaluate ALL data first
        tester.load_data(PARQUET_FILE)
        tester.evaluate()
        
        # 2. Plot General Stats
        tester.plot_error_distributions()
        
        # 3. Plot Filtered Stats (Example)
        # You can pass a list of IDs to filter just the plot without re-running evaluate
        # my_interesting_ids = ["segment_A", "segment_B"]
        # tester.plot_error_distributions(filter_ids=my_interesting_ids, filename_suffix="_special_group")
        
        # 4. Standard Best/Worst
        tester.plot_best_worst_segments(n=N_BEST_WORST)
        
        # 5. Maps
        tester.generate_maps(n_best_worst=N_BEST_WORST, n_random=N_MAP_RANDOM)

        # 6. Filtered Map Example
        # tester.generate_filtered_map(segment_ids=["segment_1", "segment_2"], map_name="map_special_segments")
        
    else:
        print(f"File {PARQUET_FILE} not found.")
        
        
if __name__ == "__main__":
    main_test()
