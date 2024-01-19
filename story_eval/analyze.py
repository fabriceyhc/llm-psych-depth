import numpy  as np
import pandas as pd
from agreement.utils.transform import pivot_table_frequency
from agreement.utils.kernels import linear_kernel, ordinal_kernel
from agreement.metrics import cohens_kappa, krippendorffs_alpha
from scipy.stats import spearmanr, pearsonr, zscore
from crowdkit.aggregation import (
    DawidSkene, 
    OneCoinDawidSkene, 
    GLAD, 
    MMSR, 
    MACE,
    Wawa
)


class AnnotationAnalyzer:

    def __init__(self, ):
        self.components = [
            'authenticity_score', 'empathy_score', 'engagement_score', 
            'emotion_provoking_score', 'narrative_complexity_score']

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
    
    def model_performances(self, ratings_df):
        return ratings_df.groupby(by=['model', 'human_quality'], dropna=False).mean(numeric_only=True)[self.components]

    def participant_scores(self, ratings_df):
        return ratings_df.groupby(by='participant_id', dropna=False).mean(numeric_only=True)[self.components]
    
    def story_scores(self, ratings_df):
        return ratings_df.groupby(by='story_id', dropna=False).mean(numeric_only=True)[self.components]
    
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


    
if __name__ == "__main__":

    # Create an instance of the class
    analyzer = AnnotationAnalyzer()

    # Read data from a CSV file 
    human_ratings_df = pd.read_csv('./human_study/data/processed/human_annotations.csv', encoding='cp1252')
    llm_ratings_df   = pd.read_csv('./human_study/data/processed/llm_annotations.csv', encoding='cp1252')

    human_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])
    llm_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])

    print(analyzer.model_performances(human_ratings_df))
    print(analyzer.participant_scores(human_ratings_df))
    print(analyzer.story_scores(human_ratings_df))

    aggregators = [MeanAggregator(), ZScoreAggregator(), DawidSkene(), OneCoinDawidSkene(), GLAD(), MMSR(), MACE(), Wawa()]
    components = ['authenticity_score', 'empathy_score', 'engagement_score', 
                 'emotion_provoking_score', 'narrative_complexity_score']
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
                human_vs_llm_corr = analyzer.comparative_correlation(human_ratings_df, llm_ratings_df, component, aggregator)
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

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("./story_eval/iaa_results.csv", index=False)