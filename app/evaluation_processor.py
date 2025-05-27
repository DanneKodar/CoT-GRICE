
import re
import os
import pandas as pd
from tqdm import tqdm


from data_handler import detect_task_type, parse_mcq_choice_number
from model_interaction import get_model_response
from openai import OpenAI 

def process_dialogs(dialogs_data, current_config, results_dir, start_iteration, openai_client):
    """Processes all dialogs and tasks, returning a list of results."""

    results = []
    iterations = 0
    max_iterations = current_config.get('max_iterations', float('inf'))
    prompt_style = current_config.get('model', {}).get('prompt_style', 'cot').lower() 
    start_iteration_0_based = max(0, start_iteration - 1) 
    print(f"--- Start Iteration Info ---")
    print(f"Command line start_iteration (1-based): {start_iteration}")
    print(f"Effective start index (0-based): {start_iteration_0_based}")
    print(f"Tasks will be processed starting from index {start_iteration_0_based}.")
    print(f"--------------------------")
    

    total_estimated_iterations = 0
    for dialog in dialogs_data:
        total_estimated_iterations += len(dialog.get('dialog', [])) 
        total_estimated_iterations += len(dialog.get('question', []))
    effective_max_iterations = min(max_iterations, total_estimated_iterations)

    progress_bar = tqdm(total=effective_max_iterations, desc=f"Processing Tasks ({prompt_style})", unit="task")

    for dialog_info in dialogs_data:
        if iterations >= effective_max_iterations:
            break

        dialog_id = dialog_info['dialog_id']
        full_dialog_history_text = "" 

        mcq_tasks = dialog_info.get('dialog', [])
        for turn_index, turn_data in enumerate(mcq_tasks):
            if iterations >= effective_max_iterations: break
            task_type = detect_task_type(turn_data)

            if task_type != 'MCQ':
                 print(f"Warning: Expected MCQ task, found {task_type}. Skipping.")
                 continue

            iterations += 1
            progress_bar.update(1)

            current_iteration_0_based = iterations - 1 

            if current_iteration_0_based < start_iteration_0_based:
                progress_bar.set_postfix_str(f"Skipping task {iterations}/{effective_max_iterations} (idx {current_iteration_0_based})")
                
                current_q_skipped = turn_data.get('question', '[Fråga saknas]')
                current_a_skipped = turn_data.get('answer', '[Svar saknas]')
                full_dialog_history_text += f"Question: {current_q_skipped}\nAnswer: {current_a_skipped}\n\n" 
                continue 
            progress_bar.set_postfix_str(f"Processing Dialog {dialog_id}, MCQ Turn {turn_index} (Task {iterations})")

            current_q = turn_data.get('question', '[Fråga saknas]')
            current_a = turn_data.get('answer', '[Svar saknas]')
            options_list = turn_data.get('option', [])
            correct_index = turn_data.get('answer_index', -1)
            explicit_a = turn_data.get('explict_answer', 'N/A')

            current_turn_text = f"Last Question: {current_q}\nLast Answer: {current_a}" 
            options_text = "\n".join([f"{i+1}) {opt}" for i, opt in enumerate(options_list)])

            if prompt_style == 'cot':
                prompt_text = f"""Consider the dialogue context below, paying close attention to the final Question-Answer pair.

Dialogue Context:
{full_dialog_history_text}
{current_turn_text}

Task: What is the most likely implied meaning (implicature) of the *last Answer* ("{current_a}") in response to the *last Question* ("{current_q}"), given the preceding dialogue?First, perform reasoning step-by-step based on the Dialogue Context to determine the implied meaning of the last Answer, then choose the best option (1-4) representing this implied meaning.

Options:
{options_text}

[Reasoning and answer here]:
""" 
                
            elif prompt_style == 'zero-shot':
                prompt_text = f"""Consider the dialogue context below, paying close attention to the final Question-Answer pair.

Conversation History:
{full_dialog_history_text}
{current_turn_text}

Task: What is the most likely implied meaning (implicature) of the *last Answer* ("{current_a}") in response to the *last Question* ("{current_q}"), given the preceding dialogue? Choose the best option representing this implied meaning.

Options:
{options_text}

Choice (1-4):"""
            else:
                print(f"Warning: Unknown prompt_style '{prompt_style}'. Defaulting to 'cot'.")
                prompt_text = f"""Analyze the conversation history and the current turn... [CoT prompt as above] ..."""


            messages = [{"role": "user", "content": prompt_text}]

            model_response_content, response_time, p_tokens, c_tokens, t_tokens, error = get_model_response(openai_client, messages, current_config, task_type)

            predicted_choice_number = None
            predicted_index = -1 
            is_correct = False
            reasoning = "" 

            if model_response_content and not error:
                 predicted_choice_number = parse_mcq_choice_number(model_response_content)
                 if predicted_choice_number is not None:
                     predicted_index = predicted_choice_number - 1
                     if correct_index != -1:
                         is_correct = (predicted_index == correct_index)

                 if prompt_style == 'cot':
                     reasoning_match = re.search(r'Reasoning:(.*?)Final Answer:', model_response_content, re.DOTALL | re.IGNORECASE)
                     if reasoning_match:
                         reasoning = reasoning_match.group(1).strip()
                     elif predicted_choice_number is None: 
                         reasoning = f"[Parse Error or No Final Answer] Full Response: {model_response_content}"

                 elif predicted_choice_number is None:
                     reasoning = f"[Parse Error] Full Response: {model_response_content}"


            result = {
                'dialog_id': dialog_id, 'turn_index': turn_index, 'qa_question_index': None,
                'task_type': task_type, 'question': current_q,
                'agent_answer_raw': current_a, 'options': options_list,
                'correct_index': correct_index, 'predicted_index': predicted_index,
                'predicted_choice': predicted_choice_number, 'is_correct': is_correct,
                'model_response_full': model_response_content or "", 'model_reasoning': reasoning,
                'response_time': response_time, 'prompt_tokens': p_tokens, 'completion_tokens': c_tokens,
                'total_tokens': t_tokens, 'error_type': error, 'ground_truth_answer': None,
                'prompt_style': prompt_style, 
                'full_prompt': prompt_text
            }
            results.append(result)

            full_dialog_history_text += f"Question: {current_q}\nAnswer: {current_a}\n\n"
            progress_bar.set_postfix_str(f"Dialog {dialog_id}, MCQ Turn {turn_index}")


        qa_tasks = dialog_info.get('question', [])
        for qa_index, qa_data in enumerate(qa_tasks):
            if iterations >= effective_max_iterations: break
            task_type = detect_task_type(qa_data)

            if task_type != 'QA':
                print(f"Warning: Expected QA task, found {task_type}. Skipping.")
                continue

            iterations += 1 
            progress_bar.update(1) 

            current_iteration_0_based = iterations - 1 

            if current_iteration_0_based < start_iteration_0_based:
                progress_bar.set_postfix_str(f"Skipping task {iterations}/{effective_max_iterations} (idx {current_iteration_0_based})")
                continue 
            progress_bar.set_postfix_str(f"Processing Dialog {dialog_id}, QA Task {qa_index} (Task {iterations})")
            qa_question = qa_data.get('question', '[Fråga saknas]')
            qa_ground_truth = qa_data.get('answer', '[Referenssvar saknas]')

            if prompt_style == 'zero-shot':

                qa_prompt_text = f"""Based on the conversation history provided below: Answer the question about the conversation.

    Conversation History:
    {full_dialog_history_text}
    Question: {qa_question}

    Answer:"""
            else:
                qa_prompt_text = f"""Based on the conversation history provided below and the question: First provide step-by-step reasoning to determine the answer to the question about the conversation and then write down your answer.

    Conversation History:
    {full_dialog_history_text}
    Question: {qa_question}

    Answer:"""

            messages = [{"role": "user", "content": qa_prompt_text}]

            model_response_content, response_time, p_tokens, c_tokens, t_tokens, error = get_model_response(openai_client, messages, current_config, task_type)

            result = {
                'dialog_id': dialog_id, 'turn_index': None, 'qa_question_index': qa_index,
                'task_type': task_type, 'question': qa_question, 'agent_answer_raw': None,
                'options': None, 'correct_index': None, 'predicted_index': None,
                'predicted_choice': None, 'is_correct': None,
                'model_response_full': model_response_content or "", 'model_reasoning': None,
                'response_time': response_time, 'prompt_tokens': p_tokens, 'completion_tokens': c_tokens,
                'total_tokens': t_tokens, 'error_type': error, 'ground_truth_answer': qa_ground_truth,
                'prompt_style': 'qa', 
                'qa_full_prompt': qa_prompt_text
            }
            results.append(result)
            progress_bar.set_postfix_str(f"Dialog {dialog_id}, QA Task {qa_index}")
    
        if iterations > 0 and iterations % current_config.get('save_interval', 100) == 0:
            try:
                interim_df = pd.DataFrame(results)
                interim_df = pd.DataFrame(results) 
                interim_filename = os.path.join(results_dir, f'interim_results_{iterations}.csv')
                interim_df.to_csv(interim_filename, index=False, encoding='utf-8')
                tqdm.write(f"Interim results saved to {interim_filename} ({iterations} tasks processed).")
            except Exception as e:
                 tqdm.write(f"Warning: Could not save interim results at iteration {iterations}. Error: {e}")


    progress_bar.close()
    if iterations < effective_max_iterations:
         print(f"\nWarning: Processing stopped after {iterations} iterations (max_iterations might be set or loop ended early).")
    elif iterations == 0:
         print("\nWarning: No tasks were processed. Check data file and max_iterations setting.")

    return results
