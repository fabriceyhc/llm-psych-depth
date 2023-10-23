from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from typing import List, Dict
from utils import *
from loader import *


class PlanWritePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    premise = """
    Premise: {premise}"""
    
    character_prompt = """
    Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.
    
    {format_instructions}"""
    
    plan_info = """
    Premise: {premise}

    Character Portraits:
    {characters}"""

    story_prompt = """
    Task: Write a 500-word story based on the premise and character portraits. The story should be emotionally deep and impactful."""


    class OutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text


    class CharactersOutput(BaseModel):
        character_list:   List[str] = Field(description="character list")


    def generate_character_prompts(self, prompts):

        parser = PydanticOutputParser(pydantic_object=self.CharactersOutput)

        prompts_to_run = []

        for prompt_id, prompt in enumerate(prompts):

            system_profile_prompt = SystemMessagePromptTemplate.from_template(self.premise)
            human_message_prompt = HumanMessagePromptTemplate.from_template(self.character_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                partial_variables={"format_instructions": parser.get_format_instructions()},
                template_format='jinja2',
                output_parser=parser,
            )
            _input = chat_prompt.format_messages(premise=prompt)

            prompts_to_run.append({
                "id": prompt_id,
                "premise": prompt,
                "characters_prompt": prepare_prompt_for_ui(_input)
            })

        return prompts_to_run


    def generate_story_prompts(self, planwrite_df):

        prompts_to_run = []

        for prompt_id, prompt in planwrite_df.iterrows():

            system_profile_prompt = SystemMessagePromptTemplate.from_template(self.plan_info)
            human_message_prompt = HumanMessagePromptTemplate.from_template(self.story_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                output_parser=self.OutputParser(),
            )
            _input = chat_prompt.format_messages(premise=prompt['premise'],
                                                characters=create_numbered_string(prompt['character_list']))

            new_prompt = prompt.to_dict()
            new_prompt.update({
                "story_prompt": prepare_prompt_for_ui(_input)
            })
            prompts_to_run.append(new_prompt)

        return prompts_to_run


    def prompt_llm(self, prompts):

        characters_parser = PydanticOutputParser(pydantic_object=self.CharactersOutput)

        for id, prompt in enumerate(prompts):
            character_prompt = ChatPromptTemplate.from_messages([
                ("system", self.premise),
                ("human", self.character_prompt),
            ])
            story_prompt = ChatPromptTemplate.from_messages([
                ("system", self.plan_info),
                ("human", self.story_prompt),
            ])


            chain1 = character_prompt | self.llm | characters_parser
            chain2 = {"characters": create_numbered_string(chain1)} | story_prompt | self.llm | self.OutputParser()

            output = chain2.invoke({"premise": prompt})
            print(id)
            print(output)