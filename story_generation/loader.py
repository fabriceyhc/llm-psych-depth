import os
import json
from io import StringIO
import numpy  as np
import pandas as pd


class DataLoader():
    
    def load_json_files_by_filename(self, base_dir, file_name=None):

        df_list = []

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file_name is None or file == file_name:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        try:
                            json_content = json.load(f)
                            df_list.append(json_content)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in {os.path.join(root, file)}")

        df = pd.DataFrame(df_list)

        return df

    def load_json_files(self, dir, filter_key=None):

        def clean_and_parse_json(content):
            cleaned_content = content.split("```\n")[0]
            return json.loads(cleaned_content)

        json_files = [f for f in os.listdir(dir) if f.endswith('.json')]
        if filter_key is None:
            desired_files = json_files
        else:
            desired_files = [f for f in json_files if filter_key in f]

        df_list = []

        for file in desired_files:
            filepath = os.path.join(dir, file)
            with open(filepath, 'r') as f:
                content = f.read()
                parsed_data = clean_and_parse_json(content)
                df_temp = pd.DataFrame([parsed_data])
                df_list.append(df_temp)

        df_combined = pd.concat(df_list, ignore_index=True)

        return df_combined


    def load_reddit_df(self, dir, sort_by=None):

        df = self.load_json_files_by_filename(dir, "details.json")
        if sort_by is None:
            return df
        return df.sort_values(by=sort_by)


    def load_planwrite_df(self, dir, sort_by=None):

        df = self.load_json_files(dir)
        if sort_by is None:
            return df
        return df.sort_values(by=sort_by)