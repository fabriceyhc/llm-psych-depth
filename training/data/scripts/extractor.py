import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "0"

from typing import List
import pandas as pd
import datetime
from tqdm import tqdm
import json
import textwrap
import traceback

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, root_validator

class StoryExtractorModel(BaseModel):
    # characters:  List[str]  = Field(description="List of the main characters and a one sentence background profile for each.")
    archtypes:   List[str]  = Field(description="List of the character archtypes present in the story, each briefly described in one sentence.")
    plot_points: List[str]  = Field(description="List of the key plot points present in the story, each briefly described in one sentence.")
    themes:      List[str]  = Field(description="List of the primary themes present in the story, each briefly described in one sentence.")
    key_words:   List[str]  = Field(description="List of the keywords you associate with this story (e.g. tags to support story search).")

class StoryExtractor:    
    """
    This class is responsible for loading and initializing an LLM with specified parameters. 

    Parameters:
    - model_name_or_path (str): The name or path of the model to be loaded. Default is "epfl-llm/meditron-7b".
    - revision (str): The specific model revision to use. Default is "main". 
    - max_new_tokens (int): The maximum number of new tokens to be generated in a single inference. Default is 512.
    - do_sample (bool): If True, sampling is used for generating tokens. If False, deterministic decoding is used. Default is True.
    - temperature (float): The sampling temperature for token generation. Higher values lead to more randomness. Default is 0.7.
    - top_p (float): The nucleus sampling probability. Keeps the cumulative probability for the most likely tokens to this threshold. Default is 0.95.
    - top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering. Default is 40.
    - repetition_penalty (float): The penalty applied to repeated tokens. Values >1 discourage repetition. Default is 1.1.
    - cache_dir (str): The directory where the model cache is stored. Default is "./.cache/".
        - NOTE: The default dir is stored to network attached storage (NAS) and NAS is very slow for model loading. 
                However NAS has more room for LLMs. Store locally to dramatically decrease load time. 
    """
    def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", revision="main", is_chat=False,
                 max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.95, 
                 top_k=40, repetition_penalty=1.1, cache_dir="/data1/fabricehc/impossibility-watermark/.cache",
                 num_retries=3):

        self.model_name_or_path = model_name_or_path
        self.is_chat=is_chat
        self.num_retries=num_retries

        # Initialize and load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            device_map="auto",
            trust_remote_code=False,
            revision=revision) 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, cache_dir=cache_dir)

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

        # Define the output parser
        self.output_parser = PydanticOutputParser(pydantic_object=StoryExtractorModel)

        # Define prompts
        self.profile_background = """{profile}"""

        self.eval_background = textwrap.dedent("""
            [INST]

            ** Task **: 

            - Read the provided story.
            - Answer questions about the character archtypes, plot points, themes, and keywords
            without being too specific with proper nouns. Write about the archtypes and plot points
            at a higher level of abstraction. 

            ** Story **: 
            
            {story} 

            ** Format Instructions **: 
            
            {format_instructions} 

            [/INST]
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

        # Format prompt
        if self.is_chat:
            system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
            self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        else:
            self.prompt   = self.user_prompt

        self.chain = self.prompt | self.pipe | self.output_parser

    def evaluate(self, story, **kwargs):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                pydantic_output = self.chain.invoke({"story": story})
                dict_output = pydantic_output.model_dump()
                dict_output.update({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **kwargs,
                })
                return dict_output
            except Exception:
                retry_count += 1
                print(f"Failed to produce a valid evaluation, trying again...")
                if retry_count >= self.num_retries:
                    print(f"Failed to produce a valid evaluation after {retry_count} tries.")
                    print(traceback.format_exc())

if __name__ == "__main__":

    se = StoryExtractor()

    source_stories_df = pd.read_csv("./training/data/processed/source_stories.csv")

    llm_name = se.model_name_or_path.replace("/", ".")
    save_path = f"./training/data/processed/{llm_name}_story_extractions.csv"

    try:
        df = pd.read_csv(save_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["book_id", "title", "characters", "plot_points", "themes", "key_words", "timestamp", "llm_annotator"])

    for i, story_data in tqdm(source_stories_df.iterrows(), total=source_stories_df.shape[0]):
        if df[df['book_id'] == story_data['book_id']].empty:
            response = se.evaluate(
                story=story_data['story'],
                book_id=story_data['book_id'],
                title=story_data['title'],
                llm_annotator=se.model_name_or_path
            )
            df = df._append(response, ignore_index=True)
            df.to_csv(save_path, index=False)
        else:
            print(f"Previously evaluation found. Skipping book_id={story_data['book_id']}")

    print(f"Story extractions saved to {save_path}")