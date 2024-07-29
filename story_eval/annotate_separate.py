import time
import traceback
import pandas as pd
import guidance
from guidance import models, gen, select, user, system, assistant

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

# Define annotation fn
@guidance
def annotate_psd(lm, persona, story, component):

    if "authenticity" in component:
        desc = """
        - Authenticity 
            - Does the writing feel true to real human experiences? 
            - Does it represent psychological processes in a way that feels authentic and believable?
        """
    if "emotion provoking" in component:
        desc = """
        - Emotion Provoking 
            - How well does the writing depict emotional experiences? 
            - Does it explore the nuances of the characters' emotional states, rather than just describing them in simple terms? 
            - Can the writing show rather than tell a wide variety of emotions? 
            - Do the emotions that are shown in the text make sense in the context of the story? 
        """
    if "empathy" in component:
        desc = """
        - Empathy 
            - Do you feel like you were able to empathize with the characters and situations in the text? 
            - Do you feel that the text led you to introspection, or to new insights about yourself or the world?" 
        """
    if "engagement" in component:
        desc = """
        - Engagement 
            - Does the text engage you on an emotional and psychological level? 
            - Do you feel the need to keep reading as you read the text? 
        """
    if "narrative complexity" in component:
        desc = """
        - Narrative Complexity 
            - Do the characters in the story have multifaceted personalities? Are they developed beyond stereotypes or tropes? Do they exhibit internal conflicts? 
            - Does the writing explore the complexities of relationships between characters? 
            - Does it delve into the intricacies of conflicts and their partial or complete resolutions? 
        """
    if "human likeness" in component:
        desc = """
        - Human Likeness 
            - Does the story appear to be written by a human or an AI? 
        """

    id_ = component.replace(" ", "_")

    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Task Description: 
        1. Review the given component of psychological depth: {component}. Be sure to understand component and the questions that characterize it.
        2. Read a given story, paying special attention to the degree to which '{component}' is present in the story.
        3. Concisely explain the degree to which '{component}' is evident in the story in a single sentence. 
        4. Assign a rating for '{component}' from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).

        ### Description of Psychological Depth Components:  
        
        We illustrate this component of sychological depth in terms the following questions: 
        {desc}
 
        ### The story to evaluate:
        {story}
        """
    with assistant():
        lm += f"""\
        {component} Explanation: {gen(f'{id_}_explanation', stop='.')}.
        {component} Score: {gen(f'{id_}_score', regex='[1-5]')}
        """
    return lm

model_ids = [
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-70B-Instruct", 
    "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ", 
    "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ",
]

# dataset = pd.read_csv("/data2/fabricehc/llm-psych-depth/data/study_stories.csv")
# dataset = dataset[dataset['round'] == 1]
# assert len(dataset) == 100

# story_id,premise_id,premise,text,author_type,author_short,author_full,net_upvotes
dataset = pd.read_csv("/data2/fabricehc/llm-psych-depth/data/study_stories.csv", encoding='8859')
dataset = dataset[dataset["round"] == 1]

components = [
    "authenticity",
    "emotion provoking",
    "empathy",
    "engagement",
    "narrative complexity",
    "human likeness",
]

use_personas = False

personas = [
    "You are a helpful AI who specializes in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
    "You are a helpful AI who focuses on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
    "You are a helpful AI who evaluates how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
    "You are a helpful AI who examines the text for its ability to provoke a wide range of intense emotional responses in the reader.",
    "You are a helpful AI who analyzes the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
] if use_personas else [""]

for model_id in model_ids:

    save_path = f"/data2/fabricehc/llm-psych-depth/human_study/data/processed/{model_id.replace('/', '--')}_separate_annotations.csv"

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

    for participant_id, persona in enumerate(personas):

        for index, row in dataset.iterrows():
            # Skip rows that have already been annotated
            if not existing_annotations.empty and ((existing_annotations["participant_id"] == participant_id) & (existing_annotations["story_id"] == row["story_id"]) & (existing_annotations["premise_id"] == row["premise_id"])).any():
                print(f"Skipping already annotated row: participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}")
                continue
            
            story = row["text"]

            annotation = {}
            start_time = time.time()
            for component in components:
                id_ = component.replace(" ", "_")
                try:
                    output = llm + annotate_psd(persona=persona, story=story, component=component)
                    output_dict = {
                        f'{id_}_explanation': output[f'{id_}_explanation'],
                        f'{id_}_score': output[f'{id_}_score'],
                    }
                    annotation.update(output_dict)
                except Exception:
                    print(traceback.format_exc())
                    print(f"Error on: participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}")

            time_taken = time.time() - start_time
            annotation.update({
                "participant_id": participant_id, 
                "persona": persona, 
                "time_taken": time_taken,
                **row,
            })
            print(f"Results for participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}': {annotation}")
            results.append(annotation)

            # Append new results to existing annotations and save to CSV
            df = pd.DataFrame(results)
            combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["participant_id", "story_id", "premise_id"])
            combined_df.to_csv(save_path, index=False)

    # Delete previous llm to free up memory asap
    del llm