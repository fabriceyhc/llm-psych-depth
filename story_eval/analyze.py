import os
import numpy as np
import pandas as pd
from agreement.utils.transform import pivot_table_frequency
from agreement.utils.kernels import linear_kernel, ordinal_kernel
from agreement.metrics import cohens_kappa, krippendorffs_alpha
from scipy.stats import spearmanr, pearsonr, zscore, ttest_ind, f_oneway

from crowdkit.aggregation import (
    DawidSkene, 
    OneCoinDawidSkene, 
    GLAD, 
    MACE,
    Wawa
)

class AnnotationAnalyzer:

    def __init__(self, ):
        self.components = [
            'authenticity_score', 'empathy_score', 'engagement_score', 
            'emotion_provoking_score', 'narrative_complexity_score', 'human_likeness_score']

    def regular_iaa(self, ratings_df, component, prefix="human"):
            
        questions_answers_table = pivot_table_frequency(ratings_df["story_id"], ratings_df[component])
        users_answers_table = pivot_table_frequency(ratings_df["participant_id"], ratings_df[component])

        # unweighted
        unweighted_cohens_kappa = cohens_kappa(questions_answers_table, users_answers_table)
        unweighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table)
        # linear weighted
        linear_weighted_cohens_kappa = cohens_kappa(questions_answers_table, users_answers_table, weights_kernel=linear_kernel)
        linear_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table, weights_kernel=linear_kernel)
        # ordinal weighted
        ordinal_weighted_cohens_kappa = cohens_kappa(questions_answers_table, users_answers_table, weights_kernel=ordinal_kernel)
        ordinal_weighted_krippendorffs_alpha = krippendorffs_alpha(questions_answers_table, weights_kernel=ordinal_kernel)
        
        # spearman
        spearman = self.pairwise_iaa(ratings_df, component, reduce=True)

        results = {
            f"component": component,
            f"{prefix}_unweighted_cohens_kappa": unweighted_cohens_kappa,
            f"{prefix}_unweighted_krippendorffs_alpha": unweighted_krippendorffs_alpha,
            f"{prefix}_linear_weighted_cohens_kappa": linear_weighted_cohens_kappa,
            f"{prefix}_linear_weighted_krippendorffs_alpha": linear_weighted_krippendorffs_alpha,
            f"{prefix}_ordinal_weighted_cohens_kappa": ordinal_weighted_cohens_kappa,
            f"{prefix}_ordinal_weighted_krippendorffs_alpha": ordinal_weighted_krippendorffs_alpha,
        }

        results.update(spearman)

        return results
    
    def pairwise_iaa(self, ratings_df, component, reduce=False):

        cols = ['story_id', 'participant_id', component]
        df = ratings_df[cols]
        
        pivot_table = df.pivot(index='story_id', columns='participant_id', values=component)
        rater_ids = pivot_table.columns
        iaa_corrs = []
        for i in range(len(rater_ids)):
            for j in range(i+1, len(rater_ids)):
                corr, p = spearmanr(pivot_table[rater_ids[i]], pivot_table[rater_ids[j]])
                iaa_corrs.append({
                    "component": component,
                    "pairwise_spearman_corr": corr,
                    "pairwise_spearman_p_value": p,
                    "participant_id_1": rater_ids[i],
                    "participant_id_2": rater_ids[j]
                })

        df = pd.DataFrame(iaa_corrs)
        if reduce:
            df_grouped = df.groupby(by="component").mean()
            return {
                "pairwise_spearman_corr": df_grouped["pairwise_spearman_corr"].item(),
                "pairwise_spearman_p_value": df_grouped["pairwise_spearman_p_value"].item(),
            }
        return df
    
    def aggregate_consensus_labels(self, ratings_df, component, aggregator=MACE()):
        cols = ['participant_id', 'story_id', component]
        df = ratings_df[cols]
        df = df.astype({component:'float'})
        df.columns = ["worker", "task", "label"]
        consensus_labels = aggregator.fit_predict(df).to_list()
        return consensus_labels

    def comparative_correlation(self, human_annotations, llm_annotations, component, aggregator=MACE()):
        human_consensus_labels = self.aggregate_consensus_labels(human_annotations, component, aggregator)
        llm_consensus_labels   = self.aggregate_consensus_labels(llm_annotations, component, aggregator)
        spearman_corr, spearman_p_value = spearmanr(human_consensus_labels, llm_consensus_labels)
        pearson_corr, pearson_p_value = pearsonr(human_consensus_labels, llm_consensus_labels)
        return {
            "component": component,
            "human_vs_llm_spearman_corr": spearman_corr,
            "human_vs_llm_spearman_p_value": spearman_p_value,
            "human_vs_llm_pearson_corr": pearson_corr,
            "human_vs_llm_pearson_p_value": pearson_p_value,
            "human_consensus_labels": human_consensus_labels,
            "llm_consensus_labels": llm_consensus_labels,
        }
    

    def model_scores_w_stdev(self, ratings_df):
        # Calculate mean and std dev for each group
        grouped = ratings_df.groupby('model_short', dropna=False)
        # Initialize an empty DataFrame to store the formatted results
        result_df = pd.DataFrame(index=grouped.indices.keys())

        # Calculate and format mean ± std for each component
        for component in self.components:
            mean_series = grouped[component].mean()
            std_series = grouped[component].std()
            result_df[component] = mean_series.map('{:.2f}'.format) + " ± " + std_series.map('{:.2f}'.format)

        result_df = result_df.reset_index().rename(columns={'index': 'model_short'}).sort_values(by=["model_short"], key=lambda x: x.map(sort_order))

        return result_df

    def model_scores(self, ratings_df):
        result_df = ratings_df.groupby(by=['model_short'], dropna=False).mean(numeric_only=True)[self.components]
        result_df = result_df.reset_index().rename(columns={'index': 'model_short'}).sort_values(by=["model_short"], key=lambda x: x.map(sort_order))
        return result_df

    def participant_scores(self, ratings_df):
        result_df = ratings_df.groupby(by='participant_id', dropna=False).mean(numeric_only=True)[self.components]
        result_df = result_df.reset_index().rename(columns={'index': 'participant_id'}).dropna()
        return result_df
    
    def story_scores(self, ratings_df):
        scores = ratings_df.groupby(by='story_id', dropna=False).mean(numeric_only=True)
        stories = ratings_df[["story_id", "model_short", "strategy"]].drop_duplicates()
        # assert (len(stories)) == 100
        return scores.merge(stories, on="story_id")
    
    def strategy_scores(self, ratings_df):
        result_df = ratings_df.groupby(by=['model_short', 'strategy'], dropna=False).mean(numeric_only=True)[self.components]
        result_df = result_df.reset_index().rename(columns={'index': 'model_strategy'}).dropna()
        return result_df
    
    def summarize_iaa_and_corr(self, ratings_df):
        cols = ["human_ordinal_weighted_krippendorffs_alpha", "llm_ordinal_weighted_krippendorffs_alpha", 
                "human_vs_llm_spearman_corr", "human_vs_llm_spearman_p_value"]
        ratings_df = ratings_df[ratings_df["excluded_participant_id"] == -1] # all raters
        ratings_df = ratings_df[ratings_df["aggregator"] == "MeanAggregator"] # only Wawa aggregation
        return ratings_df.groupby(by='component', dropna=False).mean(numeric_only=True)[cols]
    
    def calculate_binarized_accuracy(self, df):
        # Binarizing the human_likeness_score
        df['model_label'] = df['model_short'].apply(lambda x: 'human' if 'human' in x.lower() else 'llm')
        df['predicted_label'] = df['human_likeness_score'].apply(lambda x: 'human' if x in [4, 5] else ('llm' if x in [1, 2] else 'wrong'))

        # Calculate binarized accuracy
        correct_predictions = df[df['predicted_label'] == df['model_label']].shape[0]
        total_predictions = df[df['predicted_label'] != 'wrong'].shape[0]
        binarized_accuracy = correct_predictions / total_predictions if total_predictions else 0

        return binarized_accuracy
    
    def calculate_ordinal_accuracy(self, df):
        # Function to calculate penalty
        def penalty(row):
            if row['model_label'] == 'human':
                return abs(row['human_likeness_score'] - 5)
            else:  # 'llm'
                return abs(row['human_likeness_score'] - 1)

        # Calculate penalties for each row
        df['model_label'] = df['model_short'].apply(lambda x: 'human' if 'human' in x.lower() else 'llm')
        df['penalty'] = df.apply(penalty, axis=1)
        total_penalty = df['penalty'].sum()
        max_penalty = df.shape[0] * 4  # Maximum possible penalty
        penalized_accuracy = 1 - (total_penalty / max_penalty if max_penalty else 0)

        return penalized_accuracy

    def count_average_words(self, stories_df):
        stories_df["story_len"] = stories_df["text"].apply(lambda x: len(x.split()))
        stories_df = stories_df.groupby(by='model_short', dropna=False).mean(numeric_only=True)["story_len"]
        stories_df = stories_df.reset_index().rename(columns={'index': 'model_short'}).sort_values(by=["model_short"], key=lambda x: x.map(sort_order))
        stories_df = stories_df._append({"model_short": "ALL", "story_len": stories_df["story_len"].mean()}, ignore_index=True)
        return stories_df

    def measure_component_corrs(self, ratings_df):
        correlation_matrix = ratings_df[self.components].corr().reset_index().rename(columns={'index': 'component'})
        return correlation_matrix

    def perform_pairwise_ttests(self, ratings_df, col="participant_id"):
        """
        Perform pairwise t-tests for each score type among participants.

        :param ratings_df: pandas DataFrame containing the dataset
        :return: pandas DataFrame with the t-test results
        """
        unique_vals = ratings_df[col].unique()
        t_test_results = []

        for score in self.components:
            for i in range(len(unique_vals)):
                for j in range(i + 1, len(unique_vals)):
                    data_i = ratings_df[ratings_df[col] == unique_vals[i]][score]
                    data_j = ratings_df[ratings_df[col] == unique_vals[j]][score]

                    # Perform t-test between the two participants
                    t_stat, p_val = ttest_ind(data_i, data_j, nan_policy='omit')

                    # Append the result
                    t_test_results.append({
                        'score': score,
                        f'{col}_1': unique_vals[i],
                        f'{col}_2': unique_vals[j],
                        't_stat': t_stat,
                        'p_value': p_val
                    })

        df = pd.DataFrame(t_test_results)
        df.loc['Average'] = df.mean(numeric_only=True)
        df.at['Average', 'score'] = 'Average'
        return df

    def perform_anova(self, ratings_df, col="participant_id"):
        """
        Perform ANOVA to compare all participants against each other for each score type.

        :param ratings_df: pandas DataFrame containing the dataset
        :return: pandas DataFrame with the ANOVA results
        """
        unique_vals = ratings_df[col].unique()
        anova_results = []

        for score in self.components:
            score_data = [ratings_df[ratings_df[col] == pid][score] for pid in unique_vals]
            f_stat, p_val = f_oneway(*score_data)
            anova_results.append({'score': score, 'f_stat': f_stat, 'p_value': p_val})

        df = pd.DataFrame(anova_results)
        df.loc['Average'] = df.mean(numeric_only=True)
        df.at['Average', 'score'] = 'Average'
        return df
    
class ZScoreAggregator:

    def fit_predict(self, df):
        """
        Aggregates individual rater scores using z-scores and returns the aggregated scores.

        Parameters:
        - df: DataFrame with columns ['task', 'worker', 'label']

        Returns:
        - Series with aggregated rating.
        """

        # Convert labels to float type
        df['label'] = df['label'].astype(float)

        # Pivot the data to get a wide format
        wide_df = df.pivot(index='task', columns='worker', values='label')

        # Convert each rater's labels to z-scores.
        zscore_df = wide_df.apply(zscore, axis=0, result_type='broadcast')

        # Aggregate z-scores by taking the mean across raters for each item.
        aggregated_zscores = zscore_df.mean(axis=1)

        # Convert aggregated z-scores back to the original scale.
        mean_label = wide_df.stack().mean()
        std_label = wide_df.stack().std()
        aggregated_labels = (aggregated_zscores * std_label) + mean_label

        # Return results in the desired format
        result_series = pd.Series(data=aggregated_labels.values).round(decimals=2)

        return result_series
    
class MeanAggregator:

    def fit_predict(self, df):
        """
        Aggregates individual rater scores via their mean and returns the aggregated scores.

        Parameters:
        - df: DataFrame with columns ['task', 'worker', 'label']

        Returns:
        - Series with aggregated rating.
        """

        # Convert labels to float type
        df['label'] = df['label'].astype(float)

        # Aggregate the ratings by taking the mean for each task
        aggregated_labels = df.groupby('task')['label'].mean()

        # Return results in the desired format, rounding to two decimal places
        result_series = aggregated_labels.round(decimals=2)

        return result_series

def filter_out_values(df, column, values_to_filter):
    mask = ~df[column].isin(values_to_filter)
    return df[mask]
    
if __name__ == "__main__":

    llm_name = "gpt-4"

    # Custom Sort Order
    sort_order = {
        "Llama-2-7B" :        0, 
        "Llama-2-13B":        1, 
        "Vicuna-33B" :        2,
        "Llama-2-70B":        3,
        "Mixtral-8x7B":       4,
        "GPT-3.5":            5,
        "GPT-4":              6,
        "Human-Novice":       7,
        "Human-Intermediate": 8,
        "Human-Advanced":     9,
    }

    # Create an instance of the class
    analyzer = AnnotationAnalyzer()

    # Read data from a CSV file 
    human_ratings_df     = pd.read_csv('./human_study/data/processed/human_annotations.csv', encoding='cp1252')
    llm_ratings_df       = pd.read_csv(f'./human_study/data/processed/{llm_name}_annotations.csv', encoding='cp1252')
    stories_df           = pd.read_csv(f'./human_study/data/stories.csv')
    benchmark_stories_df = pd.read_csv(f'./data/study_stories.csv', encoding='8859')

    blacklist = [83, 70, 71] 
    stories_df = filter_out_values(stories_df, "story_id", blacklist)
    benchmark_stories_df = filter_out_values(benchmark_stories_df, "study_id", blacklist)

    human_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])
    llm_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])

    # cols = ["story_id", "participant_id", 'authenticity_score', 'empathy_score', 'engagement_score', 
    #             'emotion_provoking_score', 'narrative_complexity_score', "human_likeness_score"]
    # human_ratings_df[cols].to_csv(f'./story_eval/human_annotations_sorted.csv', index=False)
    # llm_ratings_df[cols].to_csv(f'./story_eval/gpt-4_annotations_sorted.csv', index=False)

    analyzer.model_scores(human_ratings_df).to_csv(f'./story_eval/tables/human_study_model_scores.csv', index=False)
    analyzer.model_scores_w_stdev(human_ratings_df).to_csv(f'./story_eval/tables/human_study_model_scores_w_stdev.csv', index=False)
    analyzer.participant_scores(human_ratings_df).to_csv(f'./story_eval/tables/human_study_participant_scores.csv', index=False)
    analyzer.story_scores(human_ratings_df).to_csv(f'./story_eval/tables/human_study_story_scores.csv', index=False)
    analyzer.strategy_scores(human_ratings_df).to_csv(f'./story_eval/tables/human_study_strategy_scores.csv', index=False)
    analyzer.count_average_words(stories_df).to_csv(f'./story_eval/tables/human_study_count_average_words.csv', index=False)
    analyzer.count_average_words(benchmark_stories_df).to_csv(f'./story_eval/tables/human_study_count_average_words_benchmark.csv', index=False)
    analyzer.measure_component_corrs(human_ratings_df).to_csv(f'./story_eval/tables/human_study_component_corrs.csv', index=False)
    analyzer.perform_pairwise_ttests(human_ratings_df, col="participant_id").to_csv(f'./story_eval/tables/human_study_pairwise_t_tests_participant_id.csv', index=False)
    analyzer.perform_pairwise_ttests(human_ratings_df, col="model_short").to_csv(f'./story_eval/tables/human_study_pairwise_t_tests_model.csv', index=False)
    analyzer.perform_anova(human_ratings_df, col="participant_id").to_csv(f'./story_eval/tables/human_study_anova_participant_id.csv', index=False)
    analyzer.perform_anova(human_ratings_df, col="model_short").to_csv(f'./story_eval/tables/human_study_anova_model.csv', index=False)
    
    save_path = f"./story_eval/tables/human_vs_{llm_name}_iaa_raw.csv"

    if not os.path.exists(save_path):

        aggregators = [MeanAggregator(), ZScoreAggregator(), DawidSkene(), OneCoinDawidSkene(), GLAD(), MACE(), Wawa()]
        components = ['authenticity_score', 'empathy_score', 'engagement_score', 
                    'emotion_provoking_score', 'narrative_complexity_score', "human_likeness_score"]
        participant_ids = [-1] + human_ratings_df["participant_id"].unique().tolist()

        results = []
        for participant_id in participant_ids:
            filtered_human_ratings_df = human_ratings_df[human_ratings_df['participant_id'] != participant_id]
            # print(f"Excluding participant_id={participant_id}...")

            # Calculate regular and comparative IAA for each category
            for component in components:
                human_iaa = analyzer.regular_iaa(filtered_human_ratings_df, component, prefix="human")
                llm_iaa   = analyzer.regular_iaa(llm_ratings_df, component, prefix="llm")
                # print(f"Component: {component}")
                # print(f"human_iaa:\n{human_iaa}")
                # print(f"llm_iaa:\n{llm_iaa}")

                for aggregator in aggregators:   
                    human_vs_llm_corr = analyzer.comparative_correlation(filtered_human_ratings_df, llm_ratings_df, component, aggregator)
                    # print(f"Aggregator: {aggregator.__class__.__name__}")
                    # print(f"human_vs_llm:\n{human_vs_llm_corr}")
                    results.append({
                        "excluded_participant_id": participant_id,
                        "component": component,
                        **human_iaa,
                        **llm_iaa,
                        "aggregator": aggregator.__class__.__name__,
                        **human_vs_llm_corr
                    })

        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(save_path, index=False)

    else:
        results_df = pd.read_csv(save_path)

    # Compute average kripp alpha and correlation (with p-value)
    analyzer.summarize_iaa_and_corr(results_df).to_csv(f'./story_eval/tables/human_vs_{llm_name}_iaa_corrs.csv')

    # Calculate accuracies
    save_path = f"./story_eval/tables/human_vs_llm_prediction_accuracies.csv"
    accuracies = []
    for model in human_ratings_df["model_short"].unique():

        human_model_df = human_ratings_df[human_ratings_df["model_short"] == model]
        llm_model_df = llm_ratings_df[llm_ratings_df["model_short"] == model]
        
        # Calculate accuracies
        human_binarized_accuracy = analyzer.calculate_binarized_accuracy(human_model_df)
        human_ordinal_accuracy = analyzer.calculate_ordinal_accuracy(human_model_df)

        llm_binarized_accuracy = analyzer.calculate_binarized_accuracy(llm_model_df)
        llm_ordinal_accuracy = analyzer.calculate_ordinal_accuracy(llm_model_df)

        accuracies.append({
            "model": model,
            "human_binarized_accuracy": human_binarized_accuracy,
            "human_ordinal_accuracy": human_ordinal_accuracy,
            f"{llm_name}_binarized_accuracy": llm_binarized_accuracy,
            f"{llm_name}_ordinal_accuracy": llm_ordinal_accuracy,
        })

    # Human Average
    human_avg_df = human_ratings_df[human_ratings_df["model_short"].str.lower().str.contains("human")]
    llm_avg_df = llm_ratings_df[llm_ratings_df["model_short"].str.lower().str.contains("human")]

    human_binarized_accuracy_havg = analyzer.calculate_binarized_accuracy(human_avg_df)
    human_ordinal_accuracy_havg = analyzer.calculate_ordinal_accuracy(human_avg_df)

    llm_binarized_accuracy_havg = analyzer.calculate_binarized_accuracy(llm_avg_df)
    llm_ordinal_accuracy_havg = analyzer.calculate_ordinal_accuracy(llm_avg_df)

    accuracies.append({
        "model": "Human",
        "human_binarized_accuracy": human_binarized_accuracy_havg,
        "human_ordinal_accuracy": human_ordinal_accuracy_havg,
        f"{llm_name}_binarized_accuracy": llm_binarized_accuracy_havg,
        f"{llm_name}_ordinal_accuracy": llm_ordinal_accuracy_havg,
    })

    # LLM
    human_avg_df = human_ratings_df[~human_ratings_df["model_short"].str.lower().str.contains("human", )]
    llm_avg_df = llm_ratings_df[~llm_ratings_df["model_short"].str.lower().str.contains("human")]

    human_binarized_accuracy_lavg = analyzer.calculate_binarized_accuracy(human_avg_df)
    human_ordinal_accuracy_lavg = analyzer.calculate_ordinal_accuracy(human_avg_df)

    llm_binarized_accuracy_lavg = analyzer.calculate_binarized_accuracy(llm_avg_df)
    llm_ordinal_accuracy_lavg = analyzer.calculate_ordinal_accuracy(llm_avg_df)

    accuracies.append({
        "model": "LLM",
        "human_binarized_accuracy": human_binarized_accuracy_lavg,
        "human_ordinal_accuracy": human_ordinal_accuracy_lavg,
        f"{llm_name}_binarized_accuracy": llm_binarized_accuracy_lavg,
        f"{llm_name}_ordinal_accuracy": llm_ordinal_accuracy_lavg,
    })

    # Overall
    human_binarized_accuracy_all = analyzer.calculate_binarized_accuracy(human_ratings_df)
    human_ordinal_accuracy_all = analyzer.calculate_ordinal_accuracy(human_ratings_df)

    llm_binarized_accuracy_all = analyzer.calculate_binarized_accuracy(llm_ratings_df)
    llm_ordinal_accuracy_all = analyzer.calculate_ordinal_accuracy(llm_ratings_df)

    accuracies.append({
        "model": "Overall",
        "human_binarized_accuracy": human_binarized_accuracy_all,
        "human_ordinal_accuracy": human_ordinal_accuracy_all,
        f"{llm_name}_binarized_accuracy": llm_binarized_accuracy_all,
        f"{llm_name}_ordinal_accuracy": llm_ordinal_accuracy_all,
    })

    accuracies_df = pd.DataFrame(accuracies).sort_values(by=["model"], key=lambda x: x.map(sort_order))
    print(accuracies_df)
    accuracies_df.to_csv(save_path, index=False)
