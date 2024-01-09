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
    MACE, # best option for our current data 
    Wawa
)


class InterAnnotatorAgreement:

    def __init__(self, ):
        pass

    def regular_iaa(self, ratings_df, component):
            
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
            "component": component,
            "unweighted_cohens_kappa": unweighted_cohens_kappa,
            "unweighted_krippendorffs_alpha": unweighted_krippendorffs_alpha,
            "linear_weighted_cohens_kappa": linear_weighted_cohens_kappa,
            "linear_weighted_krippendorffs_alpha": linear_weighted_krippendorffs_alpha,
            "ordinal_weighted_cohens_kappa": ordinal_weighted_cohens_kappa,
            "ordinal_weighted_krippendorffs_alpha": ordinal_weighted_krippendorffs_alpha,
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
                    "spearman_corr": corr,
                    "spearman_p_value": p,
                    "gid_1": rater_ids[i],
                    "gid_2": rater_ids[j]
                })

        df = pd.DataFrame(iaa_corrs)
        if reduce:
            df_grouped = df.groupby(by="component").mean()
            return {
                "spearman_corr": df_grouped["spearman_corr"].item(),
                "spearman_p_value": df_grouped["spearman_p_value"].item(),
            }
        return df
    
    def aggregate_consensus_labels(self, ratings_df, component, aggregator=MACE()):
        cols = ['participant_id', 'story_id', component]
        df = ratings_df[cols]
        df = df.astype({component:'float'})
        df.columns = ["worker", "task", "label"]
        consensus_labels = aggregator.fit_predict(df).to_list()
        return consensus_labels

    def comparative_iaa(self, ratings_df1, ratings_df2, component):
        pass
    
if __name__ == "__main__":

    # Create an instance of the class
    iaa_calculator = InterAnnotatorAgreement()

    # Read data from a CSV file or DataFrame
    ratings_df = pd.read_csv('./human_study/data/processed/annotations.csv')

    # Calculate regular and comparative IAA for each category
    components = ['authenticity_score', 'empathy_score', 'engagement_score', 
                 'emotion_provoking_score', 'narrative_complexity_score']
    # llm_ratings = ...  # Replace with your LLM ratings for each category

    for component in components:
        regular_iaa = iaa_calculator.regular_iaa(ratings_df, component)
        pairwise_iaa = iaa_calculator.pairwise_iaa(ratings_df, component)
        # comparative_iaa = iaa_calculator.comparative_iaa(category, llm_ratings[category])
        print(f"Component: {component}")
        print(regular_iaa)
        # print(pairwise_iaa)
        consensus_labels = iaa_calculator.aggregate_consensus_labels(ratings_df, component)
        print(consensus_labels)