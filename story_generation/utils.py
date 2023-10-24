import re
import os
import json


def extract_string_prompt(chat_message_prompt_list):
    return f"{chat_message_prompt_list[0].to_json()['kwargs']['content']}\n{chat_message_prompt_list[-1].to_json()['kwargs']['content']}"


def create_numbered_string(lst):
    if lst and re.match(r'^\d+\.\s', lst[0]):
        return '\n'.join(lst)

    numbered_str = ""
    for index, item in enumerate(lst, 1):
        numbered_str += f"{index}. {item}\n"
    return numbered_str.strip()


def first_n_words(s, n=5):
    words = s.split()
    if len(words) < n:
        return '_'.join(words)
    return '_'.join(words[:n])


def save_json_files(save_dir, save_info):

    os.makedirs(save_dir, exist_ok=True)

    for prompt in save_info:
        filename = f"{prompt['id']}_{first_n_words(prompt['premise'])}.json"
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(prompt, f, indent=4)