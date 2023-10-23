from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, StrOutputParser
from langchain.chains import LLMChain

from utils import *


class WriterProfilePromptsGenerator:

    def __init__(self, llm) -> None:
        self.llm = llm

    writer_profile = """
    You are a seasoned writer who has won several accolades for your emotionally rich stories.
    When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to.
    Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters.
    Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions.
    Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last."""

    story_prompt = """
    Now write a 500-word story on the following prompt:
    
    {prompt}"""


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


    def prompt_llm(self, prompts):

        for id, prompt in enumerate(prompts):
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", self.writer_profile),
                ("human", self.story_prompt),
            ])
            # chain = chat_prompt | self.llm | self.OutputParser()
            # output = chain.invoke({'prompt': prompt})
            chain = LLMChain(llm=self.llm, prompt=chat_prompt, output_parser=self.OutputParser())
            output = chain.run(prompt=prompt)
            print(id)
            print(output)