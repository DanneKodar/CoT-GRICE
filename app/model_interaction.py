
from openai import OpenAI
import time
import os


def get_model_response(client, messages, current_config, task_type):
    '''sends requests and recieves answers from the openai API'''
    model_config = current_config['model']
    default_max_tokens = model_config.get('max_tokens', 200)
    QA_max_tokens = 200

    if task_type == 'QA':
            effective_max_tokens = QA_max_tokens
    else:
        effective_max_tokens = default_max_tokens
    try:
        model_config = current_config['model']
        api_params = {
            'model': model_config['model'],
            'messages': messages,
            'max_tokens': effective_max_tokens,
            'temperature': model_config['temperature'],
            'timeout': model_config.get('timeout', 30) 
        }

        if task_type == 'MCQ' and model_config.get('use_logit_bias', False):
            logit_bias_config = model_config.get('logit_bias_map', {
                '16': 100, 
                '17': 100, 
                '18': 100, 
                '19': 100 
            })
            api_params['logit_bias'] = logit_bias_config

        start_time = time.time()
        response = client.chat.completions.create(**api_params)
        end_time = time.time()

        response_time = end_time - start_time
        model_response = response.choices[0].message.content.strip()

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        total_tokens = response.usage.total_tokens if response.usage else 0

        return model_response, response_time, prompt_tokens, completion_tokens, total_tokens, None

    except Exception as e:
        error_message = f"API Error: {str(e)}"
        print(f"\n{error_message}") 
        return None, 0, 0, 0, 0, error_message

if __name__ == '__main__':
    print("Testing model interface (requires valid config and API key)...")
