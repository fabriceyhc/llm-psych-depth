from utils import *
from loader import DataLoader
from writer_profile import WriterProfilePromptsGenerator
from plan_write import PlanWritePromptsGenerator
from custom_llm import CustomLLM
from bard_llm import BardLLM


from langchain.llms import OpenAI

def main():

    # llm = CustomLLM()
    # llm = OpenAI(openai_api_key="xxx", model_name="gpt-4")
    llm = BardLLM()

    writer_profile_generator = WriterProfilePromptsGenerator(llm=llm)
    plan_write_generator = PlanWritePromptsGenerator(llm=llm)

    loader = DataLoader()
    r_wp_dir = "../data/v3/"
    r_wp_df = loader.load_reddit_df(r_wp_dir, sort_by='prompt')
    r_prompts = r_wp_df['prompt']

    save_dir = "../llm_story_generation_results_v1_regen/"
    model_name = "bard-2023.10.30"

    num_generations = 3
    # To generate stories for all, set regen_ids to empty or None 
    writer_profile_regen_ids = [0, 1, 2]
    plan_write_regen_ids = [3, 4, 5]

    for n in range(num_generations):
        writer_profile_generator.prompt_llm(r_prompts, save_dir, model_name, writer_profile_regen_ids)
        plan_write_generator.prompt_llm(r_prompts, save_dir, model_name, plan_write_regen_ids)

    # print('### Writer Profile Story Generation Example Prompt ###\n' +
    #       '-' * 54 + '\n' +
    #       writer_profile_generator.generate_prompts(r_prompts)[0]['story_generation_prompt'] + '\n' +
    #       '=' * 54 + '\n')

    # print('### Plan + Write Characters Generation Example Prompt ###\n' +
    #       '-' * 57 + '\n' +
    #       plan_write_generator.generate_character_prompts(r_prompts)[0]['characters_prompt'] + '\n' +
    #       '=' * 57 + '\n')
    
    # planwrite_dir = "../gpt4_story_generation_results/plan_write_v2/"
    # planwrite_df = loader.load_planwrite_df(planwrite_dir, sort_by='id')

    # print('### Plan + Write Story Generation Example Prompt ###\n' +
    #       '-' * 52 + '\n' +
    #       plan_write_generator.generate_story_prompts(planwrite_df)[0]['story_prompt'] + '\n' +
    #       '=' * 52 + '\n')


if __name__ == '__main__':
    main()