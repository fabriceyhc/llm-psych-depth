import pandas as pd
import glob

###############################################################
# Helper Functions                                            #
###############################################################

def csv_to_df(csv_files, mappings):
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
    return pd.concat(dfs, axis=1)

def reform_melt(df, keep_cols, col_append="score"):

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
                            value_name=f'{rename_val}_{col_append}')
        
        df_melted = df_melted.rename(columns={participant_id_col:"participant_id"})
        
        # print(df_melted)
        # print(df_melted.columns)
        # print(df_melted.shape)
        # print(df_melted[f'{rename_val}_col'][0])
        # print(df_melted[f'{rename_val}_col'][1])
        
        # Displaying the reshaped DataFrame for component
        results.append(df_melted)

    df = pd.concat(results, axis=1)[keep_cols]
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def add_study_ids(df, divisor=None, add_in=0):
    if divisor is None:
        divisor = len(df["participant_id"].unique())
    df['study_id'] = df.index // divisor + add_in
    return df

def filter_out_values(df, column, values_to_filter):
    mask = ~df[column].isin(values_to_filter)
    return df[mask]

###############################################################
# Load Raw Data                                               #
###############################################################

csv_files = glob.glob("./human_study/data/preprocessed/Story Annotation Study *")
STUDY_1_files = [f for f in csv_files if "[2024]" not in f]
STUDY_2_files = [f for f in csv_files if "[2024]" in f]

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

# convert separate csv files into a single dataframe
STUDY_1_df = csv_to_df(STUDY_1_files, mappings)
STUDY_2_df = csv_to_df(STUDY_2_files, mappings)

# STUDY_1_df.to_csv("./human_study/data/preprocessed/study_1_annotations_raw.csv", index=False)
# STUDY_2_df.to_csv("./human_study/data/preprocessed/study_2_annotations_raw.csv", index=False)

# filter out only the required columns and pivot so that the only columns are the participant_id and the scores 
keep_cols = ["participant_id", "authenticity_score", "empathy_score", "engagement_score", 
            "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"]
STUDY_1_df = reform_melt(STUDY_1_df, keep_cols, col_append="score")
STUDY_2_df = reform_melt(STUDY_2_df, keep_cols, col_append="score")

STUDY_1_df["human_likeness_score"] = 6 - STUDY_1_df["human_likeness_score"] # later decided to invert the scale so human == 5 and LLM == 1

# add story_id column based on the number of participants
STUDY_1_df = add_study_ids(STUDY_1_df)
STUDY_2_df = add_study_ids(STUDY_2_df, add_in=100)

# join the studies together
df = pd.concat([STUDY_1_df, STUDY_2_df], ignore_index=True)

# filter out bad stories from the analysis
# study_id=83 is a fragment story of ~2 sentences
# study_id=[45,70] are the same story, so we only need to keep the first one.
# study_id=[14,71] are the same story, so we only need to keep the first one.
blacklist = [83, 70, 71] 
df = filter_out_values(df, "study_id", blacklist)

final_cols = ["participant_id", "study_id", "authenticity_score", "empathy_score", "engagement_score", 
              "emotion_provoking_score", "narrative_complexity_score", "human_likeness_score"]

# df[final_cols].to_csv("./human_study/data/preprocessed/annotations.csv", index=False)
stories_df = pd.read_csv("./data/study_stories.csv", encoding='8859')
human_annotations_df = df[final_cols].merge(stories_df[['study_id', 'story_id', 'premise_id','model_short','strategy']], on="study_id")
human_annotations_df.to_csv("./human_study/data/processed/human_annotations.csv", index=False)

###############################################################
# Comments / Reasoning                                        #
###############################################################

mappings = [
    ("comments on authentic elements within the story (optional)", "authenticity"),
    ("comments on empathetic elements in the story (optional)", "empathy"),
    ("comments on engaging elements within the story (optional)", "engagement"),
    ("comments on emotion provoking elements in the story (optional)", "emotion_provoking"),
    ("comments on narrative complexity in the story (optional)", "narrative_complexity"),
    ("comments on how you determined human vs. llm authorship (optional)", "human_likeness"),
    ("open feedback\r\n(use this field for miscellaneous feedback on the story as a whole)", "open")
]

# convert separate csv files into a single dataframe
STUDY_1_df = csv_to_df(STUDY_1_files, mappings)
STUDY_2_df = csv_to_df(STUDY_2_files, mappings)

# filter out only the required columns and pivot so that the only columns are the participant_id and the scores 
keep_cols = ["participant_id", "authenticity_comments", "empathy_comments", "engagement_comments", 
             "emotion_provoking_comments", "narrative_complexity_comments", "human_likeness_comments", "open_comments"]
STUDY_1_df = reform_melt(STUDY_1_df, keep_cols, col_append="comments")
STUDY_2_df = reform_melt(STUDY_2_df, keep_cols, col_append="comments")

# add story_id column based on the number of participants
STUDY_1_df = add_study_ids(STUDY_1_df)
STUDY_2_df = add_study_ids(STUDY_2_df, add_in=100)

# join the studies together
df = pd.concat([STUDY_1_df, STUDY_2_df], ignore_index=True)

# filter out bad stories from the analysis
# study_id=83 is a fragment story of ~2 sentences
# study_id=[45,70] are the same story, so we only need to keep the first one.
# study_id=[14,71] are the same story, so we only need to keep the first one.
blacklist = [83, 70, 71] 
df = filter_out_values(df, "study_id", blacklist)

df.to_csv("./human_study/data/preprocessed/annotation_comments_raw.csv", index=False)

comment_cols = ["participant_id", "study_id", "authenticity_comments", "empathy_comments", "engagement_comments", 
             "emotion_provoking_comments", "narrative_complexity_comments", "human_likeness_comments", "open_comments"]

df[comment_cols].to_csv("./human_study/data/preprocessed/annotation_comments.csv", index=False)
stories_df = pd.read_csv("./data/study_stories.csv", encoding='8859')
human_annotations_df = df[comment_cols].merge(stories_df[['study_id', 'story_id', 'premise_id','model_short','strategy']], on="study_id")
human_annotations_df.to_csv("./human_study/data/processed/human_annotation_comments.csv", index=False)

###############################################################
# Combined Scores and Comments                                #
###############################################################

final_cols = [
    "study_id",
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
    "strategy", 
]

scores_df = pd.read_csv("./human_study/data/processed/human_annotations.csv")
comments_df = pd.read_csv("./human_study/data/processed/human_annotation_comments.csv")
df = pd.merge(scores_df, comments_df[comment_cols], on=["participant_id", "study_id"])[final_cols]
df.to_csv("./human_study/data/processed/human_annotations_and_comments.csv", index=False)