
import yaml
import sys
import os

def load_config(config_path='config.yaml'):
    try:

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"Loaded config from: {config_path}")
            if 'data' in config and 'path' in config['data']:
                 config['data']['path'] = config['data']['path'].replace('\\', '/')
                 print(f"Adjusted data path to: {config['data']['path']}")
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file '{config_path}': {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("Testing config loader...")
    test_config = load_config('../config.yaml') 
    if test_config:
        print("Config loaded successfully.")
        print(test_config)