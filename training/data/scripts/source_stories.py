import re
import pandas as pd

def remove_lines_containing(text, phrase):
    phrase_lower = phrase.lower()
    return '\n'.join(line for line in text.split('\n') if phrase_lower not in line.lower())

def reduce_consecutive_newlines(text):
    return re.sub(r'\n{3,}', '\n\n', text)

def remove_last_line(text):
    lines = text.split('\n')
    if lines and len(lines) > 1:
        return '\n'.join(lines[:-1])
    else:
        return ''  # Return an empty string if there's only one line or no text

def token_estimate(text):
    words = text.split()
    return round(len(words) * 1.25)

# class TokenCounter:
#     def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-v0.1-GPTQ"):
#         from transformers import AutoTokenizer
#         self.model_name_or_path = model_name_or_path
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

#     def __call__(self, text):
#         tokens = self.tokenizer.tokenize(text)
#         return len(tokens)

if __name__ == "__main__":

    stories_df = pd.read_csv("./training/data/preprocessed/stories.zip")
    metadata_df = pd.read_csv("./training/data/preprocessed/db_books.csv")
    df = pd.merge(stories_df, metadata_df, on=["bookno"])

    name_mapping = {
        "bookno": "book_id",
        "content": "story",
        "Title": "title",
        "Author": "author",
        "Language": "language"
    }

    df = df.rename(columns=name_mapping)

    df['token_estimate'] = df['story'].apply(token_estimate) # TokenCounter()
    df['language']   = df['language'].str.strip()

    # ROW FILTERS
    max_tokens = 16000
    allowable_languages = ["English"]

    df = df[df['token_estimate'] <= max_tokens]
    df = df[df['language'].isin(allowable_languages)]

    # CONTENT FILTERS

    strings_to_remove = [
        "PROJECT GUTENBERG",
        "Produced by ",
        "Online Distributed Proofreading Team",
        "http://www.pgdp.net",
        "http://www.pgdpcanada.net", 
        "THE END"
    ]

    for string in strings_to_remove:
        df['story'] = df['story'].apply(lambda x: remove_lines_containing(x, string))

    df['story'] = df['story'].apply(reduce_consecutive_newlines)
    df['story'] = df['story'].str.strip()
    df['story'] = df['story'].apply(remove_last_line)
    df['story'] = df['story'].str.strip()

    print(df)

    df.to_csv("./training/data/processed/source_stories.csv", index=False)
