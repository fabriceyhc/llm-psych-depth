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

    def comparative_correlation(self, human_annotations, llm_annotations, component, aggregator=MACE()):
        human_consensus_labels = iaa_calculator.aggregate_consensus_labels(human_annotations, component, aggregator)
        llm_consensus_labels   = iaa_calculator.aggregate_consensus_labels(llm_annotations, component, aggregator)
        corr, p = spearmanr(human_consensus_labels, llm_consensus_labels)
        return {
            "component": component,
            "spearman_corr": corr,
            "spearman_p_value": p,
            "human_consensus_labels": human_consensus_labels,
            "llm_consensus_labels": llm_consensus_labels,
        }
    
if __name__ == "__main__":

    # Create an instance of the class
    iaa_calculator = InterAnnotatorAgreement()

    # Read data from a CSV file 
    human_ratings_df = pd.read_csv('./human_study/data/processed/human_annotations.csv', encoding='cp1252')
    llm_ratings_df   = pd.read_csv('./human_study/data/processed/llm_annotations.csv', encoding='cp1252')

    human_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])
    llm_ratings_df.sort_values(['participant_id', 'story_id'], ascending=[True, True])

    aggregators = [DawidSkene(), OneCoinDawidSkene(), GLAD(), MMSR(), MACE(), Wawa()]
    components = ['authenticity_score', 'empathy_score', 'engagement_score', 
                 'emotion_provoking_score', 'narrative_complexity_score']

    # Calculate regular and comparative IAA for each category
    for component in components:
        human_iaa = iaa_calculator.regular_iaa(human_ratings_df, component)
        llm_iaa = iaa_calculator.regular_iaa(llm_ratings_df, component)
        print(f"Component: {component}")
        print(f"human_iaa:\n{human_iaa}")
        print(f"llm_iaa:\n{llm_iaa}")

        for aggregator in aggregators:   
            print(f"Aggregator: {aggregator.__class__.__name__}")
            human_vs_llm_corr = iaa_calculator.comparative_correlation(human_ratings_df, llm_ratings_df, component, aggregator)
            print(f"human_vs_llm:\n{human_vs_llm_corr}")