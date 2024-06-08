import pandas as pd
import datetime
from tqdm import tqdm
import json
import time
import textwrap
import traceback
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.llms.fake import FakeListLLM # just for testing...
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class PsychDepthEval(BaseModel):
    authenticity_explanation:         str   = Field(description="explanation of authenticity score")
    authenticity_score:               float = Field(description="degree to which the writing is authentic (1 is Implausible - 5 is Undeniably Real)")
    emotion_provoking_explanation:    str   = Field(description="explanation of emotion provoking score")
    emotion_provoking_score:          float = Field(description="degree to which the writing is emotion provoking (1 is Unmoving - 5 is Highly Emotional)")
    empathy_explanation:              str   = Field(description="explanation of empathy score")
    empathy_score:                    float = Field(description="degree to which the writing is empathetic (1 is Detached - 5 is Deep Resonance)")
    engagement_explanation:           str   = Field(description="explanation of engagement score")
    engagement_score:                 float = Field(description="degree to which the writing is engaging (1 is Unengaging - 5 is Captivating)")
    narrative_complexity_explanation: str   = Field(description="explanation of narrative complexity score")
    narrative_complexity_score:       float = Field(description="degree to which the writing is narratively complex (1 is Simplistic - 5 is Intricately Woven)")
    human_likeness_explanation:       str   = Field(description="explanation of whether the story is human or LLM written")
    human_likeness_score:             float = Field(description="likelihood that the story is human or LLM written  (1 is Very Likely LLM - 5 is Very Likely Human)")

class StoryEvaluator:
    def __init__(self, openai_model="gpt-4", test_mode=True, num_retries=10):

        self.openai_model = openai_model
        self.num_retries=num_retries

        self.output_parser = PydanticOutputParser(pydantic_object=PsychDepthEval)

        # self.persona_background = textwrap.dedent("""      
        #     {persona}
        #     Your expertise lies in the study of psychological depth in literature.
        #     Your reputation is built on your ability to assess writing with both precision and fairness.
        #     You aren't easily swayed by superficial charm and always prioritize substance over style.
        #     You believe very strongly that the only way to be kind and compassionate to a writer is
        #     to provide honest and constructive feedback, especially when there is room for improvement.
        #     Offer feedback that is candid and honest, but also constructive.
        #     """
        # )
        self.persona_background = "{persona}"

        self.eval_background = textwrap.dedent("""

            ###Task Description: 
            1. Review the given components of psychological depth: authenticity, emotion provoking, empathy, engagement, and narrative complexity. Be sure to understand each concept and the questions that characterize them.
            2. Read a given story, paying special attention to components of psychological depth.
            3. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is average and 5 is greatly above average (should be rare to provide this score).
            4. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely LLM written and 5 means very likely human written. 

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

            ###Format Instructions: 
            
            {format_instructions} 
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.persona_background,
            input_variables=["persona"],
        )

        self.user_prompt = PromptTemplate(
            template=self.eval_background,
            input_variables=["story"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        # format chat prompt
        system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
        user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
        self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        if test_mode:
            fake_output = {
                "authentic_explanation": "The writing displays genuine emotions and thoughts.",
                "authentic_score": 4.3,
                "emotion_provoking_explanation": "The writing effectively evokes strong feelings.",
                "emotion_provoking_score": 3.6,
                "empathy_explanation": "The writing shows a deep understanding of others' feelings.",
                "empathy_score": 4.6,
                "engagement_explanation": "The writing captivates the reader's attention throughout.",
                "engagement_score": 4.0,
                "narrative_complexity_explanation": "The narrative structure is intricate and layered.",
                "narrative_complexity_score": 3.7,
                "human_or_llm_explanation": "The writing seems too stilted to be human.",
                "human_or_llm_score": 5
            }
            responses=[f"Here's what I think:\n{json.dumps(fake_output)}" for i in range(2)]
            self.llm = FakeListLLM(responses=responses)
        else:
            load_dotenv(find_dotenv()) # load openai api key from ./.env
            self.llm = ChatOpenAI(model_name=self.openai_model)
            self.base_temperature = self.llm.temperature

        self.chain = self.prompt | self.llm | self.output_parser

    def evaluate(self, persona, story, **kwargs):
        retry_count = 0
        while retry_count < self.num_retries:
            if retry_count == 0 and self.llm.temperature != self.base_temperature:
                print(f"Resetting base temperature back to {self.base_temperature}")
                self.llm.temperature = self.base_temperature
            try:
                pydantic_output = self.chain.invoke({"persona": persona, "story": story})
                dict_output = pydantic_output.model_dump()
                dict_output.update({
                    "persona": persona, 
                    "story": story,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": self.llm.temperature, 
                    **kwargs
                })
                return dict_output
            except Exception:
                retry_count += 1
                self.llm.temperature += 0.1
                print(f"Failed to produce a valid evaluation. Changing temperature to {self.llm.temperature} and trying again...")
                if retry_count >= self.num_retries:
                    print(f"Failed to produce a valid evaluation after {retry_count} tries. Reseting temperature and skipping problem story: \n{story}")
                    print(traceback.format_exc())
                    self.llm.temperature = self.base_temperature

if __name__ == "__main__":

    openai_model = "gpt-4o-2024-05-13" # "gpt-3.5-turbo-0125"
    use_mop = True

    se = StoryEvaluator(openai_model=openai_model, test_mode=False)

    if use_mop:
        personas = [
            "You are a helpful AI who specializes in evaluating the genuineness and believability of characters, dialogue, and scenarios in stories.",
            "You are a helpful AI who focuses on identifying and assessing moments in the narrative that effectively evoke empathetic connections with the characters.",
            "You are a helpful AI who evaluates how well a story captures and maintains the reader's interest through pacing, suspense, and narrative flow.",
            "You are a helpful AI who examines the text for its ability to provoke a wide range of intense emotional responses in the reader.",
            "You are a helpful AI who analyzes the structural and thematic intricacy of the plot, character development, and the use of literary devices.",
        ]
    else:
        personas = [""]

    dataset = pd.read_csv("/data2/fabricehc/llm-psych-depth/data/study_stories.csv", encoding='8859')
    dataset = dataset[dataset["round"] == 1]
    dataset = dataset[dataset["study_id"] != 70]
    dataset = dataset[dataset["study_id"] != 71]
    dataset = dataset[dataset["study_id"] != 83]

    if use_mop:
        save_path = f'./human_study/data/processed/{openai_model}_mop_annotations.csv'
    else:
        save_path = f'./human_study/data/processed/{openai_model}_no_mop_annotations.csv'

    # Check if the save file already exists and load it
    try:
        existing_annotations = pd.read_csv(save_path)
    except FileNotFoundError:
        existing_annotations = pd.DataFrame()

    results = []
    for participant_id, persona in enumerate(personas):
        for index, row in dataset.iterrows():
            # Skip rows that have already been annotated
            if not existing_annotations.empty and ((existing_annotations["participant_id"] == participant_id) & (existing_annotations["story_id"] == row["story_id"]) & (existing_annotations["premise_id"] == row["premise_id"])).any():
                print(f"Skipping already annotated row: participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}")
                continue

            start_time = time.time()
            output_dict = se.evaluate(
                persona=persona, 
                story=row['text'],
            )
            time_taken = time.time() - start_time
            output_dict.update({
                "participant_id": participant_id, 
                "persona": persona, 
                "time_taken": time_taken,
                **row,
            })
            print(f"Results for participant_id={participant_id}, story_id={row['story_id']}, premise_id={row['premise_id']}': {output_dict}")
            results.append(output_dict)

            # Append new results to existing annotations and save to CSV
            df = pd.DataFrame(results)
            combined_df = pd.concat([existing_annotations, df]).drop_duplicates(subset=["participant_id", "story_id", "premise_id"])
            combined_df.to_csv(save_path, index=False)

    print(f"Story ratings saved to {save_path}")
