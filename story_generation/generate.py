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
from pydantic import BaseModel, Field

from utils import *
from loader import *
from writer_profile import *
from plan_write import *


class PlanWritePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    premise = "Premise: {premise}\n"
    
    character_prompt = "Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.\n\n{format_instructions}"
    
    plan_info = "Premise: {premise}\n\nCharacter Portraits:\n{characters}\n"

    story_prompt = "Task: Write a 500-word story based on the premise and character portraits. The story should be emotionally deep and impactful."
    
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

    # public UI url: https://e86faac610b22eab21.gradio.live
    URI = 'wss://log-assessed-degree-substitute.trycloudflare.com/api/v1/stream'
    llm = CustomLLM(URI=URI)


    loader = DataLoader()
    r_wp_dir = "data/v3/"
    r_wp_df = loader.load_reddit_df(r_wp_dir, sort_by='prompt')
    r_prompts = r_wp_df['prompt']
    
    
    writer_profile_generator = WriterProfilePromptsGenerator(llm=llm)

    print('### Writer Profile Story Generation Example Prompt ###\n' +
          '-' * 54 + '\n' +
          writer_profile_generator.generate_prompts(r_prompts)[0]['story_generation_prompt'] + '\n' +
          '=' * 54 + '\n')

    planwrite_dir = "gpt4_story_generation_results/plan_write_v2/"
    planwrite_df = loader.load_planwrite_df(planwrite_dir, sort_by='id')

    plan_write_generator = PlanWritePromptsGenerator(llm=llm)

    print('### Plan + Write Characters Generation Example Prompt ###\n' +
          '-' * 57 + '\n' +
          plan_write_generator.generate_character_prompts(r_prompts)[0]['characters_prompt'] + '\n' +
          '=' * 57 + '\n')
    
    print('### Plan + Write Story Generation Example Prompt ###\n' +
          '-' * 52 + '\n' +
          plan_write_generator.generate_story_prompts(planwrite_df)[0]['story_prompt'] + '\n' +
          '=' * 52 + '\n')


if __name__ == '__main__':
    main()