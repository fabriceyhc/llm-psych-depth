import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"

import datetime
import traceback
import textwrap
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class AuthorshipReasons(BaseModel):
    isCreative:                     bool = Field(description="Frequent mentions of creativity, original ideas, unique concepts, and imaginative storytelling.")
    isDeep:                         bool = Field(description="References to the story's ability to evoke emotions, understanding of human feelings, and emotional intelligence.")
    isNuanced:                      bool = Field(description="Discussions on the nuanced exploration of complex themes like human relationships, philosophical concepts, or societal issues.")
    isHumorous:                     bool = Field(description="The writing is described as funny or humorous.")
    isInformal:                     bool = Field(description="The writing is described as casual, informal, or using slang.")
    isUngrammatical:                bool = Field(description="Errors in punctuation and grammar.")
    hasAgressiveness:               bool = Field(description="Identification of cursing or aggressive language.")
    hasAdvancedVocab:               bool = Field(description="The writing is described as using advanced vocabulary and complex sentence structures.")
    hasAdvancedLirararyTechniques:  bool = Field(description="The writing describes the use of literary techniques like alliteration, anaphora, or metaphors.")
    hasUniqueTwists:                bool = Field(description="The presence of unexpected plot twists or unique story concepts.")
    isRepetitive:                   bool = Field(description="Repetitive use of certain phrases or words, and a lack of imaginative elements were seen as indicative of AI authorship.")
    isSimplistic:                   bool = Field(description="Perceptions of the narrative being too straightforward, predictable, or lacking in depth.")
    isRobotic:                      bool = Field(description="The writing has a robotic feel, lack of stylistic variance, and overly consistent writing style.")
    isFormulaic:                    bool = Field(description="Views that the story follows a strict structure or formula, possibly indicating an AI's patterned approach.")
    hasLowPromptAdherence:          bool = Field(description="Does not address all parts of the prompt.")

class AnnotatorEvaluator:
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
    def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", revision="main",
                 max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.95, 
                 top_k=40, repetition_penalty=1.1, cache_dir="/data1/fabricehc/impossibility-watermark/.cache",
                 num_retries=10, use_system_profile=False,):

        self.model_name_or_path = model_name_or_path
        self.use_system_profile = use_system_profile
        self.num_retries = num_retries

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
        self.output_parser = PydanticOutputParser(pydantic_object=AuthorshipReasons)

        # self.chain = self.prompt | self.pipe | self.output_parser


    def evaluate(self, input_text, profile=None, **kwargs):
        retry_count = 0
        while retry_count < self.num_retries:
            try:
                dict_input = {"input_text": input_text}
                if self.use_system_profile and profile is not None:
                    dict_input.update({"profile": profile})
                pydantic_output = self.chain.invoke(dict_input)
                dict_output = pydantic_output.model_dump()
                dict_output.update({
                    "input_text": input_text,
                    "profile": profile,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **kwargs
                })
                return dict_output
            except Exception:
                retry_count += 1
                print(f"Failed to produce a valid evaluation, trying again...")
                if retry_count >= self.num_retries:
                    print(f"Failed to produce a valid evaluation after {retry_count} tries.")
                    print(traceback.format_exc())

class ZeroShotAnnotationEvaluator(AnnotatorEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Define prompts
        self.profile_background = """{profile}"""

        self.eval_background = textwrap.dedent("""
            [INST]
            ** Task ** 

            - Review the annotation commend describing whether a short story was written by a human or by an artificial intelligence. 
            - For each possible reason, determine if the annotation comment is relateted to that reason. 

            Example:

            Annotation: 
            
            "The advanced vocabulary and use of literary techniques such as alliteration, anaphora, metaphors not only 
            make the story more interesting but show a level of complexity that I feel makes the story more likely to be written by a human."

            Answer: 
            {
                "isCreative": False,                   
                "isDeep": False,                       
                "isNuanced": True,                    
                "isHumorous": False,                   
                "isInformal": False,                   
                "isUngrammatical": False,             
                "hasAgressiveness": False,             
                "hasAdvancedVocab": True,             
                "hasAdvancedLirararyTechniques": True,
                "hasUniqueTwists": False,              
                "isRepetitive": False,                 
                "isSimplistic": False,                 
                "isRobotic": False,                    
                "isFormulaic": False,                  
                "hasLowPromptAdherence": False,        
            }

            ** Annotation Comment ** 
            
            {{input_text}}

            ** Format Instructions ** 
            
            {{format_instructions}} 
            [/INST]
            """
        )

        self.system_prompt = PromptTemplate(
            template=self.profile_background,
            input_variables=["profile"],
        )

        self.user_prompt = PromptTemplate(
            template=self.eval_background,
            input_variables=["input_text"],
            template_format='jinja2',
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

        # Format prompt
        if self.use_system_profile:
            system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
            self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        else:
            self.prompt   = self.user_prompt

        self.chain = self.prompt | self.pipe | self.output_parser

if __name__ == "__main__":

    import pandas as pd
    from tqdm import tqdm

    annotator = ZeroShotAnnotationEvaluator()
    
    #######################################################
    # Zero-shot Without Profile                           #
    #######################################################

    inputs = pd.read_csv("./human_study/data/processed/human_likeness_annotations.csv")

    llm_name = annotator.model_name_or_path.replace("/", ".")
    save_path = f"./human_study/data/preprocessed/{llm_name}_zero_shot_annotation_classifications.csv"

    df = pd.DataFrame()

    for i, input_ in tqdm(inputs.iterrows(), total=inputs.shape[0]):
        response = annotator.evaluate(
            input_text=input_['human_likeness_comments'],
            participant_id=input_['participant_id'],
            story_id=input_['story_id'],
            human_likeness_score=input_['human_likeness_score'],
            model_short=input_['model_short'],
            author_type=input_['author_type'],
            llm_annotator=annotator.model_name_or_path
        )
        df = df._append(response, ignore_index=True)
        df.to_csv(save_path, index=False)

    print(f"Reason annotations saved to {save_path}")