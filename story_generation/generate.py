import pandas as pd
from tqdm import tqdm

from writer_profile import ZeroShotWriterProfileGenerator
from plan_write import TwoStepPlanWriteGenerator


if __name__ == '__main__':

    llm = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

    writer_profile_generator = ZeroShotWriterProfileGenerator(model_name_or_path=llm)
    plan_write_generator = TwoStepPlanWriteGenerator(model_name_or_path=llm)

    premises = pd.read_csv("../data/premises.csv")

    save_dir = "../llm_story_generation_results_v2/"

    # the number of stories to be generated per prompt
    n_gen = 3
    
    # To generate stories for all, set regen_ids to empty or None 
    wp_regen_ids = [15, 16, 17, 18, 19]
    pw_regen_ids = [15, 16, 17, 18, 19]

    # min. story length
    min_len = 400

    writer_profile_generator.output_stories(premises, save_dir, llm, n_gen, wp_regen_ids, min_len)
    plan_write_generator.output_stories(premises, save_dir, llm, n_gen, pw_regen_ids, min_len)