import textwrap
import datetime
import traceback
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from story_generation.base import LLMBase

class Characters(BaseModel):
    character_list:   List[str] = Field(description="List of deep and engaging characters related to the premise. Each character should be a string in the form of <name>:<description>")

class PlanWrite(LLMBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.output_parser = StrOutputParser()
        self.characters_output_parser = PydanticOutputParser(pydantic_object=Characters)
        
        # Define prompts
        self.characters_template = textwrap.dedent(
        """
        Premise: {premise}

        Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.
        
        Format Instructions:
        {format_instructions}
        """)

        self.story_template = textwrap.dedent(
        """
        Premise: {premise}

        Character Portraits: 
        {characters}

        Task: Write a {num_words}-word story based on the premise and character portraits. The story should be emotionally deep and impactful.
        Only respond with the story.
        """)
        

        # Format prompt
        self.character_prompt = PromptTemplate(
            template=self.characters_template,
            input_variables=["premise"],
            partial_variables={"format_instructions": self.characters_output_parser.get_format_instructions()},
        )
        self.characters_chat_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate(prompt=self.character_prompt)])

        self.story_prompt = PromptTemplate(
            template=self.story_template,
            input_variables=["premise", "characters", "num_words"],
        )
        self.story_chat_prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate(prompt=self.story_prompt)])

        # Prepare chains
        self.characters_chain = self.characters_chat_prompt | self.pipe | self.characters_output_parser
        self.characters_chain_no_validation = self.characters_chat_prompt | self.pipe | self.output_parser
        self.story_chain = self.story_chat_prompt | self.pipe | self.output_parser


    def generate_characters(self, premise):
        dict_input = {"premise": premise}
        retry_count = 0
        while retry_count < self.cfg.generation_args.num_retries:
            try:
                pydantic_output = self.characters_chain.invoke(dict_input)
                characters = pydantic_output.dict()["character_list"]
                dict_output = {
                    "characters": "\n".join(f"- {c}" for c in characters),
                    "characters_retry_count": retry_count
                }
                return dict_output
            except:
                print(f"Failed to produce valid character portraits, trying again...")
                retry_count += 1
                if retry_count >= self.cfg.generation_args.num_retries:
                    print(f"Failed to produce valid character portraits {retry_count} tries.")
                    print(f"Dropping format validation and trying one more time...")
        characters = self.characters_chain_no_validation.invoke(dict_input)
        dict_output = {
            "characters": characters,
            "characters_retry_count": retry_count
        }
        return dict_output

    def generate_stories(self, premise, characters, num_words):
        dict_input = {
            "premise": premise,
            "characters": characters,
            "num_words": num_words,
        }
        story_text = self.story_chain.invoke(dict_input)
        return story_text

    # overwrite BaseLLM.generate
    def generate(self, premise, **kwargs):
        retry_count, length_retry_count = 0, 0
        while retry_count < self.cfg.generation_args.num_retries:
            try:
                characters_dict = self.generate_characters(premise)
                characters = characters_dict["characters"]
                story_text = self.generate_stories(premise, characters, num_words=self.cfg.generation_args.num_words)
                
                if not self.is_valid_length(story_text):
                    print(f"Generation length: {len(story_text.split())}")
                    print(f"Generation is not within acceptable word count range: {self.cfg.generation_args.acceptable_word_count_range}. Trying again...")
                    length_retry_count += 1
                    continue

                dict_output = {
                    "premise": premise,
                    "text": story_text,
                    "characters": characters,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "story_retry_count": retry_count,
                    "characters_retry_count": characters_dict["characters_retry_count"],
                    "length_retry_count": length_retry_count,
                    **kwargs
                }
                return dict_output
            except Exception:
                retry_count += 1
                print(f"Failed to produce a valid generation, trying again...")
                print(traceback.format_exc())
                if retry_count >= self.cfg.generation_args.num_retries:
                    print(f"Failed to produce a valid generation after {retry_count} tries.")