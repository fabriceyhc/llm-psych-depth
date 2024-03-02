import textwrap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from story_generation.base import LLMBase


class WriterProfile(LLMBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        self.output_parser = StrOutputParser()
        
        # Define prompts
        self.writer_profile = textwrap.dedent("""
        {profile}
        """)

        self.story_prompt = textwrap.dedent(
        """
        Now write a {num_words}-word story on the following prompt:
            
        {premise}

        Only respond with the story.
        """)

        self.system_prompt = PromptTemplate(
            template=self.writer_profile,
            input_variables=["profile"],
        )

        self.user_prompt = PromptTemplate(
            template=self.story_prompt,
            input_variables=["premise", "num_words"],
        )

        # Format prompt
        system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
        user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
        self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

        self.chain = self.prompt | self.pipe | self.output_parser