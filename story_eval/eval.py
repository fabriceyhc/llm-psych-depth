import numpy as np
import pandas as pd
import textwrap
from dotenv import load_dotenv, find_dotenv
from langchain.llms.fake import FakeListLLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import json


class PsychDepthEval(BaseModel):
    authentic_explanation:            str   = Field(description="explanation of authenticity score")
    authentic_score:                  float = Field(description="degree to which the writing is authentic")
    emotion_provoking_explanation:    str   = Field(description="explanation of emotion provoking score")
    emotion_provoking_score:          float = Field(description="degree to which the writing is emotion provoking")
    empathy_explanation:              str   = Field(description="explanation of empathy score")
    empathy_score:                    float = Field(description="degree to which the writing is empathetic")
    engagement_explanation:           str   = Field(description="explanation of engagement score")
    engagement_score:                 float = Field(description="degree to which the writing is engaging")
    narrative_complexity_explanation: str   = Field(description="explanation of narrative complexity score")
    narrative_complexity_score:       float = Field(description="degree to which the writing is narratively complex")

class StoryEvaluator:
    def __init__(self):

        # load api key from ./.env
        load_dotenv(find_dotenv())
        # self.llm = ChatOpenAI()
        fake_output = {
            "authentic_explanation": "The writing displays genuine emotions and thoughts.",
            "authentic_score": 8.5,
            "emotion_provoking_explanation": "The writing effectively evokes strong feelings.",
            "emotion_provoking_score": 7.2,
            "empathy_explanation": "The writing shows a deep understanding of others' feelings.",
            "empathy_score": 9.1,
            "engagement_explanation": "The writing captivates the reader's attention throughout.",
            "engagement_score": 8.0,
            "narrative_complexity_explanation": "The narrative structure is intricate and layered.",
            "narrative_complexity_score": 7.5
        }
        responses=[f"Here's what I think:\n{json.dumps(fake_output)}" for i in range(2)]
        self.llm = FakeListLLM(responses=responses)
        self.parser = PydanticOutputParser(pydantic_object=PsychDepthEval)

        self.system_profiles = [
            "You are a renowned literary critic known for your incisive and rigorous analysis.",
            "You are a novelist renowned for weaving psychologically profound narratives.", 
            "You are a literary therapist, someone who uses literature as a medium for healing and introspection.",
            "You are an experienced psychologist with a keen interest in literature.",
            "You are a professor teaching a course on psychological literature.",
        ]

        self.profile_background = textwrap.dedent("""      
            {{profile}}
            Your expertise lies in the study of psychological depth in literature.
            Your reputation is built on your ability to assess writing with both precision and fairness.
            You aren't easily swayed by superficial charm and always prioritize substance over style.
            You believe very strongly that the only way to be kind and compassionate to a writer is
            to provide honest and constructive feedback, especially when there is room for improvement.
            Offer feedback that is candid and honest, but also constructive.
            """
        )
        self.task = textwrap.dedent("""
            Your task is composed of the following steps:
            1. Review the given components of psychological depth: authenticity, emotion
            provoking, empathy, engagement, and narrative complexity. Be sure to understand
            each concept and the questions that characterize them.
            2. Read a given story, paying special attention to components of psychological depth.
            3. Think step by step and explain the degree to which each component of psychological
            depth is evident in the story.
            4. Assign a rating for each component from 1 to 5. 1 is greatly below average, 3 is
            average and 5 is greatly above average (should be rare to provide this score).
            """
        )
        self.component_desc = textwrap.dedent("""
            For our purposes, psychological depth is composed of the following concepts, each illustrated by several questions:

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
            """
        )
        self.prompt = textwrap.dedent("""
            **Task**:
            
            {task}

            **Evaluation Components**:
            
            {component_desc}

            **Story Content**:
            
            {story}

            **Format Instructions**:
            
            {format_instructions}
            """
        )
    
    def compile_prompt(self, story, profile_id=1):
        
        system_profile = self.profile_background.replace("{{profile}}", self.system_profiles[profile_id])
        system_profile_prompt = SystemMessagePromptTemplate.from_template(system_profile)
        story_prompt = HumanMessagePromptTemplate.from_template(self.prompt)
        chat_prompt = ChatPromptTemplate(
            messages=[system_profile_prompt, story_prompt],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template_format='jinja2',
            output_parser=self.parser,
        )
        prompt = chat_prompt.format_messages(task=self.task,
                                             component_desc=self.component_desc,
                                             story=story)
        return prompt
    

    def evaluate(self, story):
        chain = self.llm | self.parser
        output = chain.invoke(self.compile_prompt(story))
        return output

if __name__ == "__main__":
    se = StoryEvaluator()
    print(se.evaluate("This is a story all about how my"))