import time
import traceback
import pandas as pd
import guidance
from guidance import models, gen, select

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

# Define annotation fn
@guidance
def annotate_psd(lm, story):
    lm += f"""\
    ###System: 
    Your expertise lies in the study of psychological depth in literature.
    Your reputation is built on your ability to assess writing with both precision and fairness.
    You aren't easily swayed by superficial charm and always prioritize substance over style.
    You believe very strongly that the only way to be kind and compassionate to a writer is
    to provide honest and constructive feedback, especially when there is room for improvement.
    Offer feedback that is candid and honest, but also constructive.

    ###Task Description: 
    1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
    2. Read a given story, paying special attention to components of psychological depth.
    3. Think step by step and explain the degree to which each component of psychological depth is evident in the story.
    4. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
    5. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

    ###Description of Psychological Depth Components:  
    
    We define sychological depth in terms of the following concepts, each illustrated by several questions: 

    - Authenticity 
        - Does the writing feel true to real human experiences? 
        - Does it represent psychological processes in a way that feels authentic and believable? 
    - Emotion Provoking 
        - How well does the writing depict emotional experiences? 
        - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
        - Can the writing show rather than tell a wide variety of emotions? 
        - Do the emotions that are shown in the text make sense in the context of the story? 
    - Empathy 
        - Do you feel like you were able to empathize with the characters and situations in the text? 
        - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
    - Engagement 
        - Does the text engage you on an emotional and psychological level? 
        - Do you feel the need to keep reading as you read the text? 
    - Narrative Complexity 
        - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
        - Does the writing explore the complexities of relationships between characters? 
        - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 

    ###The story to evaluate:
    {story}

    ###Feedback: 
    ```json
    {{
        "Authenticity Score": {gen('Authenticity Score', regex='[1-5]', stop=',')},
        "Emotion Provoking Score": {gen('Emotion Provoking Score', regex='[1-5]', stop=',')},
        "Empathy Score": {gen('Empathy Score', regex='[1-5]', stop=',')},
        "Engagement Score": {gen('Engagement Score', regex='[1-5]', stop=',')},
        "Narrative Complexity Score": {gen('Narrative Complexity Score', regex='[1-5]', stop=',')},
        "Human Likeness Score": {gen('Human Likeness Score', regex='[1-5]', stop=',')},
    }}```"""
    return lm

model_ids = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]

dataset = pd.read_csv("/data2/fabricehc/llm-psych-depth/data/study_stories.csv", encoding='8859')

keys = [
    "Authenticity Score",
    "Emotion Provoking Score",
    "Empathy Score",
    "Engagement Score",
    "Narrative Complexity Score",
    "Human Likeness Score",
]

for model_id in model_ids:

    save_path = f"/data2/fabricehc/llm-psych-depth/human_study/data/processed/{model_id.replace('/', '--')}_study_annotations.csv"

    # Check if the save file already exists and load it
    try:
        existing_annotations = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_annotations = pd.DataFrame()

    # Load the model
    llm = models.Transformers(
        model_id, 
        echo=False,
        cache_dir="/data2/.shared_models/", 
        device_map='auto'
    )

    results = []
    for index, row in dataset.iterrows():
        # story_id,premise_id,premise,text,author_type,model_short,net_upvotes,strategy,characters,round,study_id

        # Skip rows that have already been annotated
        if not existing_annotations.empty and ((existing_annotations["story_id"] == row["story_id"]) & (existing_annotations["premise_id"] == row["premise_id"])).any():
            print(f"Skipping already annotated row: story_id={row['story_id']}, premise_id={row['premise_id']}")
            continue

        story = row["text"]
        try:
            start_time = time.time()
            output = llm + annotate_psd(story=story)
            time_taken = time.time() - start_time
            output_dict = extract_dict(output, keys)
            output_dict.update({
                "time_taken": time_taken,
                **row,
            })
            print(f"Results for story_id={row['story_id']}, premise_id={row['premise_id']}': {output_dict}")
            results.append(output_dict)
        except Exception:
            print(traceback.format_exc())
            print(f"Error on: story_id={row['story_id']}, premise_id={row['premise_id']}, story={story}")

        # Append new results to existing annotations and save to CSV
        df = pd.DataFrame(results)
        combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["story_id", "premise_id"])
        combined_df.to_csv(save_path, index=False)

    # Delete previous llm to free up memory asap
    del llm