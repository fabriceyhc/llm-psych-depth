import os
import json
from io import StringIO
import numpy  as np
import pandas as pd
from typing import List, Dict

from langchain.prompts.chat import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from utils import *


class DataLoader():
    
    def load_json_files_by_filename(self, base_dir, file_name=None):

        df_list = []

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file_name is None or file == file_name:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        try:
                            json_content = json.load(f)
                            df_list.append(json_content)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in {os.path.join(root, file)}")

        df = pd.DataFrame(df_list)

        return df

    def load_json_files(self, dir, filter_key=None):

        def clean_and_parse_json(content):
            cleaned_content = content.split("```\n")[0]
            return json.loads(cleaned_content)

        json_files = [f for f in os.listdir(dir) if f.endswith('.json')]
        if filter_key is None:
            desired_files = json_files
        else:
            desired_files = [f for f in json_files if filter_key in f]

        df_list = []

        for file in desired_files:
            filepath = os.path.join(dir, file)
            with open(filepath, 'r') as f:
                content = f.read()
                parsed_data = clean_and_parse_json(content)
                df_temp = pd.DataFrame([parsed_data])
                df_list.append(df_temp)

        df_combined = pd.concat(df_list, ignore_index=True)

        return df_combined


    def load_reddit_df(self, dir, sort_by=None):

        df = self.load_json_files_by_filename(dir, "details.json")
        if sort_by is None:
            return df
        return df.sort_values(by=sort_by)


    def load_planwrite_df(self, dir, sort_by=None):

        df = self.load_json_files(dir)
        if sort_by is None:
            return df
        return df.sort_values(by=sort_by)


class WriterProfilePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    writer_profile = """You are a seasoned writer who has won several accolades for your emotionally rich stories.
    When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to.
    Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters.
    Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions.
    Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.
    """

    story_prompt = """Now write a 500-word story on the following prompt:

    {prompt}
    """
    
    class OutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text


    def generate_prompts(self, prompts):

        prompts_to_run = []

        for prompt_id, prompt in enumerate(prompts):
            system_profile_prompt = SystemMessagePromptTemplate.from_template(self.writer_profile)
            human_message_prompt = HumanMessagePromptTemplate.from_template(self.story_prompt)
            chat_prompt = ChatPromptTemplate(
                messages=[system_profile_prompt, human_message_prompt],
                output_parser=self.OutputParser(),
            )
            _input = chat_prompt.format_messages(prompt=prompt)
            prompts_to_run.append({
                "prompt_id": prompt_id,
                "reddit_prompt": prompt,
                "story_generation_prompt": prepare_prompt_for_ui(_input)
            })

        return prompts_to_run


    def prompt(self, prompts):

        for prompt in prompts:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.writer_profile),
                ("human", self.story_prompt),
            ])
            chain = chat_prompt | self.llm | self.OutputParser()
            chain.invoke({"prompt": prompt})


class PlanWritePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    premise = """Premise: {premise}
    """
    
    character_prompt = """Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.
    
    {format_instructions}
    """
    
    plan_info = """Premise: {premise}

    Character Portraits:
    {characters}
    """

    story_prompt = """Task: Write a 500-word story based on the premise and character portraits. The story should be emotionally deep and impactful.
    """
    
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


    def prompt(self, wp_df):

        characters_parser = PydanticOutputParser(pydantic_object=self.CharactersOutput)

        for prompt in wp_df["prompt"]:
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

            chain2.invoke({"premise": prompt})


def main():

    from custom_llm import CustomLLM

    loader = DataLoader()
    r_wp_dir = "data/v3/"
    r_wp_df = loader.load_reddit_df(r_wp_dir, sort_by='prompt')
    r_prompts = r_wp_df['prompt']

    # public UI url: https://e86faac610b22eab21.gradio.live
    URI = 'wss://log-assessed-degree-substitute.trycloudflare.com/api/v1/stream'
    llm = CustomLLM(URI=URI)
    
    writer_profile_generator = WriterProfilePromptsGenerator(llm=llm)

    print('### Writer Profile Story Generation Example Prompt ###\n',
          '-' * 50 + '\n',
          writer_profile_generator.generate_prompts(r_prompts)[0]['story_generation_prompt'],
          '=' * 50 + '\n')

    planwrite_dir = "gpt4_story_generation_results/plan_write_v2/"
    planwrite_df = loader.load_planwrite_df(planwrite_dir, sort_by='id')

    plan_write_generator = PlanWritePromptsGenerator(llm=llm)

    print('### Plan + Write Characters Generation Example Prompt ###\n',
          '-' * 50 + '\n',
          plan_write_generator.generate_character_prompts(r_prompts)[0]['characters_prompt'],
          '=' * 50 + '\n')
    
    print('### Plan + Write Story Generation Example Prompt ###\n',
          '-' * 50 + '\n',
          plan_write_generator.generate_story_prompts(planwrite_df)[0]['story_prompt'],
          '=' * 50 + '\n')

if __name__ == '__main__':
    main()