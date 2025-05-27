
import json
import re
import sys

def load_data_from_json(filepath):
    """Loads dialogue data from the specified JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'data' not in data or 'dialogs' not in data['data']:
             raise ValueError("JSON structure is missing 'data' or 'data.dialogs' key.")
        dialogs_data = data['data']['dialogs']
        print(f"Loaded {len(dialogs_data)} dialogs from {filepath}")
        return dialogs_data
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)
    except ValueError as ve:
        print(f"Error in data file structure: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")
        sys.exit(1)


def detect_task_type(item):
    """Detects if the task is MCQ or QA based on presence of 'option' key."""
    if isinstance(item, dict) and 'option' in item and 'answer_index' in item:
        return 'MCQ'
    elif isinstance(item, dict) and 'question' in item and 'answer' in item and 'option' not in item:
        return 'QA'
    else:
        return 'Unknown'

def parse_mcq_choice_number(response_text):
    """Extracts the single digit choice (1, 2, 3, 4) from the model response."""
    if not isinstance(response_text, str):
        return None

    match = re.search(r'Final Answer:\s*([1-4])', response_text)
    if match:
        return int(match.group(1))

    match = re.search(r'[^1-4]*([1-4])[^1-4]*$', response_text)
    if match:
        return int(match.group(1))

    if len(response_text) == 1 and response_text in ['1', '2', '3', '4']:
        return int(response_text) 

    return None


if __name__ == '__main__':
    print("Testing data handler functions...")
    mcq_item = {"question": "q", "answer": "a", "explict_answer": "ea", "option": ["1","2","3","4"], "answer_index": 0}
    qa_item = {"question": "q", "answer": "a"}
    unknown_item = {"key": "value"}

    print(f"MCQ item type: {detect_task_type(mcq_item)}")
    print(f"QA item type: {detect_task_type(qa_item)}")
    print(f"Unknown item type: {detect_task_type(unknown_item)}")

    print(f"Parsing 'Final Answer: 3': {parse_mcq_choice_number('Final Answer: 3')}")
    print(f"Parsing 'blabla 2': {parse_mcq_choice_number('blabla 2')}")
    print(f"Parsing '1': {parse_mcq_choice_number('1')}")
    print(f"Parsing 'Answer is B': {parse_mcq_choice_number('Answer is B')}")
    print(f"Parsing None: {parse_mcq_choice_number(None)}")
