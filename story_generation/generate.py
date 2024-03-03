import os
import datetime
import hydra
import logging
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)

class StoryGenerator:
    def __init__(self, cfg):

        from story_generation.writer_profile import WriterProfile
        from story_generation.plan_write import PlanWrite

        self.cfg = cfg

        # Load premises to guide story generation
        self.premises = pd.read_csv(self.cfg.premises_path)
        if isinstance(self.cfg.generation_args.premise_ids, list):
            # Filter the generation for only the provided premise_ids
            self.premises = self.premises[self.premises["premise_id"].isin(self.cfg.generation_args.premise_ids)]

        if self.cfg.generation_args.strategy == "writer_profile":
            self.generator = WriterProfile(self.cfg)
            self.profile = self.cfg.writer_profile.profile
        elif self.cfg.generation_args.strategy == "plan_write":
            self.generator = PlanWrite(self.cfg)
            self.profile = ""
        else:
            raise ValueError("Must provide either 'writer_profile' or 'plan_write' for generation_args.strategy!")

        # Setup metadata for saving results
        self.save_cols = [
            "story_id",
            "premise_id",
            "premise",
            "text", 
            "author_type",
            "model_name", 
            "strategy", 
            "timestamp",
        ]

        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.save_path = self.cfg.save_path.replace(
            self.cfg.generator_args.model_name_or_path, 
            self.cfg.generator_args.model_name_or_path.replace("/", ".")
        )

    def generate(self):        
        try:
            df = pd.read_csv(self.save_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=self.save_cols)

        for i, premise in tqdm(self.premises.iterrows(), total=self.premises.shape[0]):

            if df[(df['premise_id'] == premise['premise_id']) 
                & (df['model_name'] == self.cfg.generator_args.model_name_or_path) 
                & (df['strategy'] == self.cfg.generation_args.strategy)].empty:

                story_count = 0
                while story_count < self.cfg.generation_args.num_stories:

                    response = self.generator.generate(
                        premise_id=premise['premise_id'],
                        premise=premise['premise'],
                        num_words=self.cfg.generation_args.num_words,
                        model_name=self.cfg.generator_args.model_name_or_path,
                        strategy=self.cfg.generation_args.strategy,
                        author_type="LLM",
                        profile=self.profile,
                    )

                    response.update({
                        "story_id": str(hash(response["text"]))
                    })
                    df = df.append(response, ignore_index=True)
                    df.to_csv(self.save_path, index=False)
                    story_count += 1
            else:
                log.info(f"Previously generation found. Skipping premise_id={premise['premise_id']}, model_name={self.cfg.generator_args.model_name_or_path}, and strategy={self.cfg.generation_args.strategy}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_visible_devices)
    os.environ["WORLD_SIZE"] = str(len(str(cfg.cuda_visible_devices).split(",")))
    
    g = StoryGenerator(cfg)
    g.generate()
                
if __name__ == "__main__":
    # from langchain.globals import set_debug
    # set_debug(True)
    main()
        