import os
import datetime
import traceback
import textwrap
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Dict


class CharactersOutput(BaseModel):
    character_list:   List[str] = Field(description="character list")


class PlanWriteGenerator:

    def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", revision="main",
                 max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95, 
                 top_k=40, repetition_penalty=1.1, cache_dir="/data1/fabricehc/impossibility-watermark/.cache",
                 num_retries=3, use_system_profile=True):

        self.model_name_or_path = model_name_or_path
        self.use_system_profile = use_system_profile
        self.num_retries = num_retries
        self.strategy = "plan_write"

         # Initialize and load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=False,
            revision=revision
        ) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir
        )

        # Store the pipeline configuration
        self.pipeline_config = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }

        # Create the pipeline
        pipe = pipeline("text-generation", **self.pipeline_config)
        self.pipe = HuggingFacePipeline(pipeline=pipe)

        # Define the output parsers
        self.characters_output_parser = PydanticOutputParser(pydantic_object=CharactersOutput)
        self.story_output_parser = StrOutputParser()


    def prompt_llm(self, premise, min_len=400, **kwargs):
    
        retry_count = 0
        
        while retry_count < self.num_retries:
            try:
                characters_output = self.character_chain.invoke({"premise": premise})
                dict_input = {"premise": premise, "characters": characters_output}
                output = self.story_chain.invoke(dict_input)
                story_len = len(self.tokenizer.encode(output))

                # Check story length
                if story_len < min_len:
                    retry_count += 1
                    print(f"Generated {story_len} (< {min_len}) words. Reprompting...")
                    continue
                
                dict_output = {
                    "text": output,
                    "premise": premise,
                    "characters": characters_output,
                    **kwargs,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                print(dict_output)

                return dict_output
            
            except Exception:
                retry_count += 1
                print(f"Failed to produce a valid story, trying again...")
                print(traceback.format_exc())
                if retry_count >= self.num_retries:
                    print(f"Failed to produce a valid story after {retry_count} tries.")


    def output_stories(self, premises, save_dir, llm, n_gen=3, regen_ids=None, min_len=400):

        model_name = llm.split('/')[-1]
        save_path = os.path.join(save_dir, f"{model_name}_{self.strategy}.csv")

        stories = pd.DataFrame()
        story_id = 0

        for n_row, input_ in tqdm(premises.iterrows(), total=premises.shape[0]):
        
            if regen_ids and input_['premise_id'] not in regen_ids:
                continue

            for i in range(n_gen):

                response = self.prompt_llm(
                    premise=input_['premise'],
                    min_len=min_len,
                    premise_id=input_['premise_id'],
                    story_id=story_id,
                    model_name=model_name,
                    strategy=self.strategy,
                    author_type="LLM",
                )

                stories = stories.append(response, ignore_index=True)
                story_id += 1
        
        stories.to_csv(save_path, index=False)
        print(f"Stories saved to {save_path}")


class TwoStepPlanWriteGenerator(PlanWriteGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # Define prompts
        self.premise =\
        """
        Premise: {prompt}
        """
            
        self.characters_prompt =\
        """
        Task: Based on the premise, describe the names and details of 2-3 major characters. Focus on each character's emotional states and inner thoughts.
        Only respond with the characters' names and descriptions.
        """

        self.plan_info =\
        """
        Premise: {prompt}

        Character Portraits:
        {characters}
        """

        self.story_prompt =\
        """
        Task: Write a 500-word story based on the premise and character portraits. The story should be emotionally deep and impactful.
        Only respond with the story.
        """

        self.characters_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.premise),
            ("human", self.characters_prompt),
        ])

        self.story_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.plan_info),
            ("human", self.story_prompt),
        ])

        self.character_chain = self.characters_chat_prompt | self.pipe | self.characters_output_parser
        self.story_chain =  self.story_chat_prompt | self.pipe | self.story_output_parser


if __name__ == '__main__':

    llm = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

    generator = TwoStepPlanWriteGenerator(model_name_or_path=llm)

    premises = pd.read_csv("./data/premises.csv")

    save_dir = "../llm_story_generation_results_v2/"
    os.makedirs(save_dir, exist_ok=True)

    # the number of stories to be generated per prompt
    n_gen = 3
    
    # To generate stories for all, set regen_ids to empty or None 
    regen_ids = [15, 16, 17, 18, 19]

    # min. story length
    min_len = 400

    generator.output_stories(premises, save_dir, llm, n_gen, regen_ids, min_len)