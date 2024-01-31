import pandas as pd
import glob

csv_files = glob.glob("./human_study/data/preprocessed/Story*")

###############################################################
# Ratings                                                     #
###############################################################

mappings = [
    ("**authentic**", "authenticity"),
    ("**empathy**", "empathy"),
    ("**engaging**", "engagement"),
    ("**provoke emotion**", "emotion_provoking"),
    ("**narratively complex**", "narrative_complexity"),
    ("human or an llm", "human_likeness")
]

p_id = "Participant ID\r\n(This number is assigned to you by the study organizers. Please request for one if you have not received one. Thanks!)"
dfs = []
for i, f in enumerate(csv_files):
    df = pd.read_csv(f)
    df = df.rename(columns={p_id:f"{i}.participant_id"})
    for search_val, rename_val in mappings:
        target_columns = [col for col in df.columns if search_val in col.lower()]
        renamed_columns = [f"{i}.{col}" for col in target_columns]
        df = df.rename(columns=dict(zip(target_columns, renamed_columns)))
    df = df.sort_values(by=f"{i}.participant_id")
    df["order"] = i
    dfs.append(df)
df = pd.concat(dfs, axis=1)

# print(df)
# print(df.columns)

df.to_csv("./human_study/data/preprocessed/annotations_raw.csv", index=False)

participant_id_col = "0.participant_id"

results = []
for search_val, rename_val in mappings:

    # Extracting all variations of the 'authenticity' columns
    target_columns = [col for col in df.columns if search_val in col.lower()]
    target_df = df[target_columns + [participant_id_col]]

    # print(target_df)
    # print(target_columns)
    # print(len(target_columns))
    # print(target_df.T)
    # print(target_df.columns.to_list())
    
    # Step 2: Reshape Data
    # Melt the DataFrame to transform the multiple component columns into rows
    df_melted = pd.melt(target_df, 
                        id_vars=[participant_id_col], 
                        value_vars=target_columns, 
                        var_name=f'{rename_val}_col', 
                        value_name=f'{rename_val}_score')
    
    df_melted = df_melted.rename(columns={participant_id_col:"participant_id"})
    
    # print(df_melted)
    # print(df_melted.columns)
    # print(df_melted.shape)
    # print(df_melted[f'{rename_val}_col'][0])
    # print(df_melted[f'{rename_val}_col'][1])
    
    # Displaying the reshaped DataFrame for component
    results.append(df_melted)


keep_cols = ["participant_id", "authenticity_score", "empathy_score", "engagement_score", 
             "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"]

df = pd.concat(results, axis=1)[keep_cols]

df["human_likeness_score"] = 6 - df["human_likeness_score"] # later decided to invert the scale so human == 5 and LLM == 1

# print(df)

df = df.loc[:, ~df.columns.duplicated()]
df['story_id'] = df.index // 5

# print(df)

final_cols = ["participant_id", "story_id", "authenticity_score", "empathy_score", "engagement_score", 
              "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"]

# print(df[final_cols])

df[final_cols].to_csv("./human_study/data/preprocessed/annotations.csv", index=False)
stories_df = pd.read_csv("./human_study/data/stories.csv")
human_annotations_df = df[final_cols].merge(stories_df[['story_id','model_short','model_full', 'strategy']], on=["story_id"])
human_annotations_df.to_csv("./human_study/data/processed/human_annotations.csv", index=False)


###############################################################
# Comments / Reasoning                                        #
###############################################################


mappings = [
    ("Comments on authentic elements within the story (optional)", "authenticity"),
    ("Comments on empathetic elements in the story (optional)", "empathy"),
    ("Comments on engaging elements within the story (optional)", "engagement"),
    ("Comments on emotion provoking elements in the story (optional)", "emotion_provoking"),
    ("Comments on narrative complexity in the story (optional)", "narrative_complexity"),
    ("Comments on how you determined human vs. LLM authorship (optional)", "human_likeness"),
    ("Open Feedback\r\n(Use this field for miscellaneous feedback on the story as a whole)", "open")
]

p_id = "Participant ID\r\n(This number is assigned to you by the study organizers. Please request for one if you have not received one. Thanks!)"
dfs = []
for i, f in enumerate(csv_files):
    df = pd.read_csv(f)
    df = df.rename(columns={p_id:f"{i}.participant_id"})
    for search_val, rename_val in mappings:
        target_columns = [col for col in df.columns if search_val in col]
        renamed_columns = [f"{i}.{col}" for col in target_columns]
        df = df.rename(columns=dict(zip(target_columns, renamed_columns)))
    df = df.sort_values(by=f"{i}.participant_id")
    df["order"] = i
    dfs.append(df)
df = pd.concat(dfs, axis=1)

# print(df)
# print(df.columns)

df.to_csv("./human_study/data/preprocessed/annotation_comments_raw.csv", index=False)

participant_id_col = "0.participant_id"

results = []
for search_val, rename_val in mappings:

    # Extracting all variations of the 'authenticity' columns
    target_columns = [col for col in df.columns if search_val in col]
    target_df = df[target_columns + [participant_id_col]]

    # print(target_df)
    # print(target_columns)
    # print(len(target_columns))
    # print(target_df.T)
    # print(target_df.columns.to_list())
    
    # Step 2: Reshape Data
    # Melt the DataFrame to transform the multiple component columns into rows
    df_melted = pd.melt(target_df, 
                        id_vars=[participant_id_col], 
                        value_vars=target_columns, 
                        var_name=f'{rename_val}_col', 
                        value_name=f'{rename_val}_comments')
    
    df_melted = df_melted.rename(columns={participant_id_col:"participant_id"})
    
    # print(df_melted)
    # print(df_melted.columns)
    # print(df_melted.shape)
    # print(df_melted[f'{rename_val}_col'].to_list())
    
    # Displaying the reshaped DataFrame for component
    results.append(df_melted)


keep_cols = ["participant_id", "authenticity_comments", "empathy_comments", "engagement_comments", 
             "emotion_provoking_comments", "narrative_complexity_comments", "human_likeness_comments", "open_comments"]

df = pd.concat(results, axis=1)[keep_cols]

# print(df)

df = df.loc[:, ~df.columns.duplicated()]
df['story_id'] = df.index // 5

# print(df)

comment_cols = ["participant_id", "story_id", "authenticity_comments", "empathy_comments", "engagement_comments", 
             "emotion_provoking_comments", "narrative_complexity_comments", "human_likeness_comments", "open_comments"]

# print(df[final_cols])

df[comment_cols].to_csv("./human_study/data/preprocessed/annotation_comments.csv", index=False)
stories_df = pd.read_csv("./human_study/data/stories.csv")
human_annotations_df = df[comment_cols].merge(stories_df[['story_id','model_short','model_full', 'strategy']], on=["story_id"])
human_annotations_df.to_csv("./human_study/data/processed/human_annotation_comments.csv", index=False)


###############################################################
# Combined Scores and Comments                                #
###############################################################

final_cols = [
    "participant_id", 
    "story_id", 
    "authenticity_score", 
    "empathy_score", 
    "engagement_score", 
    "emotion_provoking_score", 
    "narrative_complexity_score", 
    "human_likeness_score", 
    "authenticity_comments", 
    "empathy_comments", 
    "engagement_comments", 
    "emotion_provoking_comments", 
    "narrative_complexity_comments", 
    "human_likeness_comments",
    "open_comments",
    "model_short", 
    "model_full", 
    "strategy", 
]

scores_df = pd.read_csv("./human_study/data/processed/human_annotations.csv")
comments_df = pd.read_csv("./human_study/data/processed/human_annotation_comments.csv")
df = pd.merge(scores_df, comments_df[comment_cols], on=["participant_id", "story_id"])[final_cols]
df.to_csv("./human_study/data/processed/human_annotations_and_comments.csv", index=False)