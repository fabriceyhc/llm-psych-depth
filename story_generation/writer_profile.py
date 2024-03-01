import os
import datetime
import traceback
import textwrap
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class WriterProfileGenerator:

    def __init__(self, model_name_or_path="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", revision="main",
                 max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95, 
                 top_k=40, repetition_penalty=1.1, cache_dir="/data1/fabricehc/impossibility-watermark/.cache",
                 num_retries=3, use_system_profile=True):

        self.model_name_or_path = model_name_or_path
        self.use_system_profile = use_system_profile
        self.num_retries = num_retries
        self.strategy = "writer_profile"
        self.profile = textwrap.dedent(
        """
        You are a seasoned writer who has won several accolades for your emotionally rich stories.
        When you write, you delve deep into the human psyche, pulling from the reservoir of universal experiences that every reader, regardless of their background, can connect to.
        Your writing is renowned for painting vivid emotional landscapes, making readers not just observe but truly feel the world of your characters.
        Every piece you produce aims to draw readers in, encouraging them to reflect on their own lives and emotions.
        Your stories are a complex tapestry of relationships, emotions, and conflicts, each more intricate than the last.
        """)

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

        # Define the output parser
        self.output_parser = StrOutputParser()


    def prompt_llm(self, premise, min_len=500, **kwargs):
    
        retry_count = 0
        
        while retry_count < self.num_retries:
            try:
                dict_input = {"premise": premise, "profile": self.profile}
                output = self.chain.invoke(dict_input)
                story_len = len(self.tokenizer.encode(output))

                # Check story length
                if story_len < min_len:
                    retry_count += 1
                    print(f"Generated {story_len} (< {min_len}) words. Reprompting...")
                    continue
                
                dict_output = {
                    "text": output,
                    "premise": premise,
                    **kwargs,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                
                print(dict_output)

                return dict_output
            
            except Exception:
                retry_count += 1
                print(f"Failed to produce a valid story, trying again...")
                if retry_count >= self.num_retries:
                    print(f"Failed to produce a valid story after {retry_count} tries.")
                    print(traceback.format_exc())


    def output_stories(self, premises, save_dir, llm, n_gen=3, regen_ids=None, min_len=500):

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
                stories.to_csv(save_path, index=False)
                story_id += 1
        
        print(f"All stories saved to {save_path}")


class ZeroShotWriterProfileGenerator(WriterProfileGenerator):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # Define prompts
        self.writer_profile = """
        {profile}
        """

        self.story_prompt =\
        """
        Now write a 500-word story on the following prompt:
            
        {premise}

        Only respond with the story.
        """

        self.system_prompt = PromptTemplate(
            template=self.writer_profile,
            input_variables=["profile"],
        )

        self.user_prompt = PromptTemplate(
            template=self.story_prompt,
            input_variables=["premise"],
            # template_format='jinja2',
        )
        
        # Format prompt
        if self.use_system_profile:
            system_prompt = SystemMessagePromptTemplate(prompt=self.system_prompt)
            user_prompt   = HumanMessagePromptTemplate(prompt=self.user_prompt)
            self.prompt   = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
        else:
            self.prompt   = self.user_prompt

        self.chain = self.prompt | self.pipe | self.output_parser


if __name__ == '__main__':

    llm = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

    generator = ZeroShotWriterProfileGenerator(model_name_or_path=llm)

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