import datetime
import traceback

# from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field

from story_generation.pipeline_builder import PipeLineBuilder

class LLMBase:
    """
    This class is responsible for loading and initializing an LLM with specified parameters.  
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Build pipeline
        self.pipe = PipeLineBuilder(self.cfg.generator_args)

        # # Define the output parser
        # self.output_parser = OutputFixingParser.from_llm(
        #     parser=PydanticOutputParser(pydantic_object=DrugUseEval),
        #     llm=self.pipe,
        #     max_retries=self.cfg.generation_args.num_formatting_retries,
        # )

        # self.chain = self.prompt | self.pipe | self.output_parser

    def is_valid_length(self, text):
        word_count = len(text.split())
        low, high = self.cfg.generation_args.acceptable_word_count_range
        return low < word_count < high

    def generate(self, premise, **kwargs):
        retry_count, length_retry_count = 0, 0
        while retry_count < self.cfg.generation_args.num_retries:
            try:
                dict_input = {"premise": premise}
                if "num_words" in kwargs:
                    dict_input.update({"num_words": kwargs.get("num_words")})
                if "profile" in kwargs:
                    dict_input.update({"profile": kwargs.get("profile")})
                story_text = self.chain.invoke(dict_input)
                
                if not self.is_valid_length(story_text):
                    print(f"Generation length: {len(story_text.split())}")
                    print(f"Generation is not within acceptable word count range: {self.cfg.generation_args.acceptable_word_count_range}. Trying again...")
                    length_retry_count += 1
                    continue

                dict_output = {
                    "premise": premise,
                    "text": story_text,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "retry_count": retry_count, 
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