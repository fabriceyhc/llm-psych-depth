import pandas as pd
import datetime
from tqdm import tqdm
import json
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
    human_or_llm_explanation:         str   = Field(description="explanation of whether the story is human or LLM written")
    human_or_llm_score:               float = Field(description="likelihood that the story is human or LLM written  (1 is Very Likely Human - 5 is Very Likely LLM)")

class StoryEvaluator:
    def __init__(self, openai_model="gpt-4", test_mode=True, num_retries=10):

        self.openai_model = openai_model
        self.num_retries=num_retries

        self.output_parser = PydanticOutputParser(pydantic_object=PsychDepthEval)

        self.profile_background = textwrap.dedent("""      
            {profile}
            Your expertise lies in the study of psychological depth in literature.
            Your reputation is built on your ability to assess writing with both precision and fairness.
            You aren't easily swayed by superficial charm and always prioritize substance over style.
            You believe very strongly that the only way to be kind and compassionate to a writer is
            to provide honest and constructive feedback, especially when there is room for improvement.
            Offer feedback that is candid and honest, but also constructive.
            """
        )

        self.eval_background = textwrap.dedent("""
            ** Task **: 
            
            Your task is composed of the following steps:
            1. Review the given components of psychological depth: authenticity, emotion
            provoking, empathy, engagement, and narrative complexity. Be sure to understand
            each concept and the questions that characterize them.
            2. Read a given story, paying special attention to components of psychological depth.
            3. Think step by step and explain the degree to which each component of psychological
            depth is evident in the story.
            4. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is
            average and 5 is greatly above average (should be rare to provide this score).
            5. Lastly, estimate the likelihood that each story was authored by a human or an LLM. Think about what human or 
            LLM writing characteristics may be. Assign a score from 1 to 5, where 1 means very likely human written and 5
            means very likely LLM written. 

            ** Psychological Depth Component Descriptions **: 
            
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
            

            ** Story Content **: 
            
            {story} 

            ** Format Instructions **: 
            
            {format_instructions} 
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.profile_background,
            input_variables=["profile"],
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

    def evaluate(self, profile, story, **kwargs):
        retry_count = 0
        while retry_count < self.num_retries:
            if retry_count == 0 and self.llm.temperature != self.base_temperature:
                print(f"Resetting base temperature back to {self.base_temperature}")
                self.llm.temperature = self.base_temperature
            try:
                pydantic_output = self.chain.invoke({"profile": profile, "story": story})
                dict_output = pydantic_output.model_dump()
                dict_output.update({
                    "profile": profile, 
                    "story": story,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    se = StoryEvaluator(test_mode=False)

    system_profiles = [
            "You are a renowned literary critic known for your incisive and rigorous analysis.",
            # "You are a novelist renowned for weaving psychologically profound narratives.", 
            "You are a literary therapist, someone who uses literature as a medium for healing and introspection.",
            # "You are an experienced psychologist with a keen interest in literature.",
            "You are a professor teaching a course on psychological literature.",
        ]
    stories = pd.read_csv("./human_study/data/stories.csv")

    save_path = './human_study/data/processed/llm_annotations.csv'

    try:
        df = pd.read_csv(save_path)
    except FileNotFoundError:
        df = pd.DataFrame(
            columns=["participant_id","story_id","profile","story","timestamp","model","strategy","human_quality","llm_annotator",
                     "authenticity_explanation","authenticity_score","emotion_provoking_explanation","emotion_provoking_score",
                     "empathy_explanation","empathy_score","engagement_explanation","engagement_score",
                     "narrative_complexity_explanation","narrative_complexity_score", 
                     "human_or_llm_explanation", "human_or_llm_score"])

    for i, story_data in tqdm(stories.iterrows(), total=stories.shape[0]):
        for profile in system_profiles:
            participant_id = system_profiles.index(profile)
            if df[(df['story_id'] == story_data['story_id']) & (df['participant_id'] == participant_id)].empty:
                response = se.evaluate(
                    profile=profile, 
                    story=story_data['text'],
                    model=story_data['model'],
                    strategy=story_data['strategy'],
                    human_quality=story_data['human_quality'],
                    participant_id=participant_id,
                    story_id=story_data['story_id'],
                    llm_annotator=se.openai_model
                )
                df = df._append(response, ignore_index=True)
                df.to_csv(save_path, index=False)
            else:
                print(f"Previously evaluation found. Skipping story_id={story_data['story_id']} and participant_id={participant_id}...")

    print(f"Story ratings saved to {save_path}")