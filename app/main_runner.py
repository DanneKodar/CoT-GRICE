import argparse
import pandas as pd
import os
import sys
import traceback
import shutil
import re
from openai import OpenAI

from config_loader import load_config
from data_handler import load_data_from_json
from evaluation_processor import process_dialogs
from analysis_reporter import analyze_results

def extract_type_code(filename: str) -> str:
    '''extracts which implicature type is in the current JSON file based on the two-letter combination in the file name'''
    match = re.search(r'impl_dial_v0\.1_([a-z]{2})\.json$', filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        print(f"Warning: Could not extract type code from filename '{filename}'. Using 'unknown_type'.")
        return "unknown_type"

def run_evaluation(config_path: str, args: argparse.Namespace):
    '''loads the configuration, handles command-line arguments (especially
    output directory and start iteration), sets up the output directory
    (handling overwrites), loads data, initializes the OpenAI client,
    runs the dialogue processing, saves the results into separate MCQ and
    QA CSV files, and finally triggers the analysis of the results'''
    
    print("--- Starting Evaluation ---")
    try:
        config = load_config(config_path)
        config['loaded_config_path'] = config_path
    except SystemExit:
        return

    cli_output_dir = args.output_dir
    config_output_dir = config.get('output', {}).get('directory', './results')

    if cli_output_dir:
        results_dir = cli_output_dir
        print(f"Using output directory from command line argument: {results_dir}")
    else:
        results_dir = config_output_dir
        print(f"Using output directory from config file: {results_dir}")

    overwrite = config.get('output', {}).get('overwrite', False)
    if overwrite and os.path.exists(results_dir):
        print(f"Overwriting existing directory: {results_dir}")
        try:
            shutil.rmtree(results_dir)
        except OSError as e:
            print(f"Error removing directory {results_dir}: {e}")
            return
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {results_dir}: {e}")
        return

    try:
        data_path = config.get('data', {}).get('path', None)
        if not data_path:
            print("Error: Data path not specified in the configuration file.")
            return
        dialogs_data = load_data_from_json(data_path)
    except SystemExit:
        return
    except Exception as e:
        print(f"Caught unexpected error during data loading: {e}")
        traceback.print_exc() 
        return

    try:
        input_filename = os.path.basename(data_path)
        type_code = extract_type_code(input_filename)
        prompt_style = config.get('model', {}).get('prompt_style', 'unknown_style').lower()
        if prompt_style == 'unknown_style':
            print("Warning: 'prompt_style' not found in config. Using 'unknown_style' in filename.")
        base_prefix = f"{type_code}_{prompt_style}"
        print(f"Using base prefix for output files: {base_prefix}")
    except Exception as e:
        print(f"Error determining file prefix: {e}. Using default prefix 'default'.")
        base_prefix = "default"

    try:
        api_key_main = os.environ.get("OPENAI_API_KEY")
        if not api_key_main:
            print("ERROR: The environment variable OPENAI_API_KEY is not set for main_runner.py.")
            sys.exit(1) 
        openai_client_main = OpenAI(api_key=api_key_main)
        print("OpenAI client initialized successfully for main_runner.")
    except Exception as e:
        print(f"Could not initialize OpenAI client for main_runner: {e}")
        sys.exit(1)

    print(f"--- Processing Dialogs (Style: {prompt_style}) ---")
    try:
        start_iteration_arg = args.start_iteration
        print(f"Starting evaluation from iteration (task number): {start_iteration_arg}")

        results_list = process_dialogs(
            dialogs_data=dialogs_data,
            current_config=config,
            results_dir=results_dir,
            start_iteration=start_iteration_arg,
            openai_client=openai_client_main
        )

        if not results_list:
            print("Warning: No results were generated from dialog processing.")
        results_df = pd.DataFrame(results_list)
        print(f"--- Processing Complete ({len(results_list)} tasks processed after start iteration) ---")

        print("--- Saving Results ---")
        mcq_results_df = results_df[results_df['task_type'] == 'MCQ'].copy()
        qa_results_df = results_df[results_df['task_type'] == 'QA'].copy()

        mcq_filename = os.path.join(results_dir, f"{base_prefix}_mcq_results.csv")
        qa_filename = os.path.join(results_dir, f"{base_prefix}_qa_results.csv")

        save_error = False
        if not mcq_results_df.empty:
            try:
                mcq_results_df.to_csv(mcq_filename, index=False, encoding='utf-8')
                print(f"MCQ results saved to: {mcq_filename}")
            except Exception as e:
                print(f"Error saving MCQ results to {mcq_filename}: {e}")
                save_error = True
        else:
            print("No MCQ results generated to save.")

        if not qa_results_df.empty:
            try:
                qa_results_df.to_csv(qa_filename, index=False, encoding='utf-8')
                print(f"QA results saved to: {qa_filename}")
            except Exception as e:
                print(f"Error saving QA results to {qa_filename}: {e}")
                save_error = True
        else:
            print("No QA results generated to save.")

        if save_error:
            print("Warning: One or more errors occurred during file saving.")

        if not results_df.empty:
            print("--- Analyzing Results ---")
            analyze_results(results_df, results_dir, base_prefix)
        else:
            print("Skipping analysis as no results were generated or saved successfully.")

        print("\n--- Evaluation Finished ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing or analysis: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline with a specified configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Path to the directory to save results. Overrides config file setting if provided.'
    )
    parser.add_argument(
        '--start_iteration',
        type=int,
        default=1,
        help='The iteration number (task number) to start processing from (1-based index). Must be 1 or greater.'
    )
    args = parser.parse_args()

    config_to_run = args.config
    if not os.path.exists(config_to_run):
        print(f"Error: Specified config file '{config_to_run}' not found.")
        sys.exit(1)

    if args.start_iteration < 1:
        print("Error: --start_iteration must be 1 or greater.")
        sys.exit(1)

    try:
        run_evaluation(config_to_run, args)
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
        sys.exit(0)
