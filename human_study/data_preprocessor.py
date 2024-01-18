import pandas as pd
import glob

csv_files = glob.glob("./human_study/data/preprocessed/Story*")

p_id = "Participant ID\r\n(This number is assigned to you by the study organizers. Please request for one if you have not received one. Thanks!)"
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df = df.rename(columns={p_id:"participant_id"})
    df = df.sort_values(by="participant_id")
    dfs.append(df)
df = pd.concat(dfs)


participant_id_col = "participant_id"

mappings = [
    ("**authentic**", "authenticity"),
    ("**empathy**", "empathy"),
    ("**engaging**", "engagement"),
    ("**provoke emotion**", "emotion_provoking"),
    ("**narratively complex**", "narrative_complexity"),
    ("human or an llm", "human_or_llm")
]


results = []
for search_val, rename_val in mappings:

    # Extracting all variations of the 'authenticity' columns
    target_columns = [col for col in df.columns if search_val in col.lower()]
    target_df = df[target_columns + [participant_id_col]]
    
    # Step 2: Reshape Data
    # Melt the DataFrame to transform the multiple component columns into rows
    df_melted = pd.melt(target_df, 
                                  id_vars=[participant_id_col], 
                                  value_vars=target_columns, 
                                  var_name=f'{rename_val}_col', 
                                  value_name=f'{rename_val}_score')
    
    # Displaying the reshaped DataFrame for component
    results.append(df_melted)


keep_cols = ["participant_id", "authenticity_score", "empathy_score", "engagement_score", 
             "emotion_provoking_score", "narrative_complexity_score", "human_or_llm_score"]

df = pd.concat(results, axis=1)[keep_cols]
df = df.loc[:, ~df.columns.duplicated()]
df['story_id'] = df.index // 5

final_cols = ["participant_id", "story_id", "authenticity_score", "empathy_score", "engagement_score", 
              "emotion_provoking_score", "narrative_complexity_score", "human_or_llm_score"]

df[final_cols].to_csv("./human_study/data/preprocessed/annotations.csv", index=False)
stories_df = pd.read_csv("./human_study/data/stories_cleaned.csv")
human_annotations_df = df[final_cols].merge(stories_df[['story_id','model','strategy','human_quality']], on=["story_id"])
human_annotations_df.to_csv("./human_study/data/processed/human_annotations.csv", index=False)