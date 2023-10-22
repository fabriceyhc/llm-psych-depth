import re

def prepare_prompt_for_ui(chat_message_prompt_list):
    return f"{chat_message_prompt_list[0].to_json()['kwargs']['content']}\n{chat_message_prompt_list[-1].to_json()['kwargs']['content']}"

def create_numbered_string(lst):
    if lst and re.match(r'^\d+\.\s', lst[0]):
        return '\n'.join(lst)

    numbered_str = ""
    for index, item in enumerate(lst, 1):
        numbered_str += f"{index}. {item}\n"
    return numbered_str.strip()