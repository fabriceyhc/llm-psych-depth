import re
import os
import json
import random
import string
from nltk.corpus import stopwords
from nltk import tokenize


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

def generate_random_id(size=6):
    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    return ''.join(random.choice(characters) for i in range(size))


def split_paragraphs(text, mode='none'):
    """
    Split a text into paragraphs.
    """
    if mode == 'none':
        return [text.strip()]
    elif mode == 'newline':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        return [s.strip() for s in text.split('\n')]
    elif mode == 'newline-filter':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        paragraphs = text.split('\n')
        return [p.strip() for p in paragraphs if len(p.split()) > 100]
    elif mode == 'sentence':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        return sum([[s.strip() for s in tokenize.sent_tokenize(t)] for t in text.split('\n')], [])
    else:
        raise NotImplementedError


def get_repetition_logit_bias(tokenizer, text, bias, bias_common_tokens=False, existing_logit_bias=None, include_upper=False):
    logit_bias = {} if existing_logit_bias is None else existing_logit_bias
    for word in text.strip().split():
        processed_word = word.strip().lower()
        tokens = tokenizer.encode(word.strip()) + \
                 tokenizer.encode(' ' + word.strip())
        if include_upper:
            tokens += tokenizer.encode(processed_word.upper()) + \
                      tokenizer.encode(' ' + processed_word.upper())
        for tok in set(tokens):
            logit_bias[tok] = bias
    if not bias_common_tokens: # don't bias against common tokens (stopwords + punc)
        for tok in get_common_tokens(tokenizer):
            if tok in logit_bias:
                del logit_bias[tok]
    return logit_bias


def get_common_tokens(tokenizer):
    sw = [w.lower() for w in stopwords.words('english')]
    token_string = ''
    for word in sw:
        token_string += ' ' + word
        token_string += ' ' + word[0].upper() + word[1:]
    token_string += string.punctuation
    return set(tokenizer.encode(token_string))


def calculate_repetition_length_penalty(generation, prompt_sentences, max_length=None, tokenizer=None, is_story=False):
    if len(generation.strip()) == 0:
        return 10
    if max_length is not None and len(tokenizer.encode(generation)) > max_length:
        return 10
    if any([s in generation for s in ['I', 'Task', 'Tasks', 'Setting', 'Settings', 'Response', 'Answer', 'Answers', 'Assignment', 'Assignments', 'Backstory', 'Outline', 'Premise', 'Prompt', 'Bonus']]): # it's repeating parts of the prompt/reverting to analysis
        return 10
    if is_story:
        if any([s.lower() in generation.lower() for s in ['\nRelevant', '\nContext', '\nText', '\n1.', '\n1)', '\nRelationship', '\nMain Character', '\nCharacter', '\nConflict', '\nPlot', 'TBA', 'POV', 'protagonist', '\nEdit ', '\nPremise', '\nChapter', '\nNote', '\nFull Text', '\nNarrative', '\n(', 'All rights reserved', '(1)', 'passage', '\nRundown', '\nQuestion', '\nDiscuss', 'The story', 'This story']]): # it's repeating parts of the prompt/reverting to analysis
            return 10
        generation_paragraphs = split_paragraphs(generation, mode='newline')
        for paragraph in generation_paragraphs:
            if len(paragraph.strip()) == 0:
                continue
            if ':' in ' '.join(paragraph.strip().split()[:10]) or paragraph.strip().endswith(':'): # there's a colon in the first few words, so it's probably a section header for some fake analysis, or ends with a colon
                return 10
        penalty = 0
        for p in prompt_sentences:
            split = p.lower().split(' ')
            for i in range(6, len(split)):
                if ' '.join(split[i-5:i]) in generation.lower(): # somewhat penalize repeated strings of 5 words or more for each prompt sentence
                    penalty += 0.3
        split = generation.lower().split(' ')
        for i in range(6, len(split)):
            if ' '.join(split[i-5:i]) in ' '.join(split[i:]): # penalize repetition within the generation itself
                penalty += 0.3