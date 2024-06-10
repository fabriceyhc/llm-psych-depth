import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import ttest_ind

# Function to add a radar chart with filled variance
def add_radar_chart_filled_variance(ax, angles, data_row, label, color, variance_fraction=0.2, line_width=3):
    mean_values = data_row.filter(like='_mean').values.flatten().tolist()
    mean_values += mean_values[:1]
    std_dev = np.sqrt(data_row.filter(like='_var').values).flatten() * variance_fraction
    std_dev = np.append(std_dev, std_dev[0])
    upper_values = [mean + std for mean, std in zip(mean_values, std_dev)]
    lower_values = [mean - std for mean, std in zip(mean_values, std_dev)]
    ax.plot(angles, mean_values, color=color, linewidth=line_width, label=label)
    ax.plot(angles, upper_values, color=color, linewidth=0)
    ax.plot(angles, lower_values, color=color, linewidth=0)
    ax.fill_between(angles, lower_values, upper_values, color=color, alpha=0.1)

def plot_spider(df, models, title, sort_order, colors, save_path=None):

    # Filter data for specific models
    filtered_data = df[df["model_short"].isin(models)]

    # Remove 'participant_id' and 'story_id' from the aggregation
    filtered_data_no_ids = filtered_data.drop(columns=['participant_id', 'story_id', 'model_short', 'strategy'])

    # Re-aggregate the data without participant and story IDs
    aggregated_filtered_data_no_ids = filtered_data_no_ids.groupby('model_short').agg(['mean', 'var'])
    aggregated_filtered_data_no_ids.columns = ['_'.join(col).strip() for col in aggregated_filtered_data_no_ids.columns.values]
    aggregated_filtered_data_no_ids = aggregated_filtered_data_no_ids.sort_values(by=["model_short"], key=lambda x: x.map(sort_order))

    # Prepare data and labels for the radar chart
    labels = [label.split('_')[0].replace('_', ' ').title() for label in aggregated_filtered_data_no_ids.columns[::2]]
    labels = [label.replace('Human', 'Human Likeness').replace('Narrative', 'Narrative Complexity').replace('Emotion', 'Emotion Provoking') for label in labels]

    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Add models to the radar chart
    for idx, (model_name, row) in enumerate(aggregated_filtered_data_no_ids.iterrows()):
        color = colors[sort_order[model_name]]
        add_radar_chart_filled_variance(ax, angles, row, model_name, color, variance_fraction=0.25)

    # Format the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=20)
    ax.set_yticks(np.arange(1, 6))
    ax.set_yticklabels(np.arange(1, 6), color="grey", size=20)
    plt.title(title, size=24, y=1.1)
    ax.legend(loc='lower center', ncol=2, fontsize=16, bbox_to_anchor=(0.5, -0.22))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_model_cdf(data, model, column, ax, color, marker):
    model_data = data[data['model_short'] == model]
    values, base = np.histogram(model_data[column], bins=np.arange(1, 7), density=True)
    cumulative = np.cumsum(values)
    
    ax.plot(base[:-1], cumulative, label=model, marker=marker, color=color)
    # ax.set_title(f'{column.replace("_", " ").replace("score", "").title()}')
    ax.set_xlabel('Scores', size=20)
    ax.set_ylabel('Cumulative Probability', size=20)
    ax.set_xticks(np.arange(1, 6))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim([0, 1])

def plot_model_cdf_smooth(data, model, column, ax, color, marker):
    model_data = data[data['model_short'] == model][column]
    sns.kdeplot(model_data, cumulative=True, ax=ax, label=model, color=color)

    # ax.set_title(f'{column.replace("_", " ").replace("score", "").title()}')
    ax.set_xlabel('Scores', size=20)
    ax.set_ylabel('Cumulative Probability', size=20)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlim(0, 6)  # Set the x-axis range
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim([0, 1])

def plot_cdf(df, models, title, sort_order, colors, smooth=True, save_path=None):
    # Defining a list of unique markers for the models
    markers = ['1', '2', '3', '4', 'D', 'v', '>', '^']

    # Filter data for specific models
    filtered_data = df[df["model_short"].isin(models)]

    # Sorting the unique models based on the provided order
    sorted_df = filtered_data.sort_values(by=["model_short"], key=lambda x: x.map(sort_order))

    # Prepare data and labels for the radar chart
    score_columns = [col for col in sorted_df.columns if col.endswith('_score')]

    # Plotting the CDF for each component, including all models in the specified order
    for column in score_columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        for model, color, marker in zip(models, colors, markers):
            if smooth:
                plot_model_cdf_smooth(sorted_df, model, column, ax, color, marker)
            else:
                plot_model_cdf(sorted_df, model, column, ax, color, marker)

        # Set plot title and layout adjustments
        # ax.set_title(f'{column.replace("_", " ").replace("score", "").title()}Ratings', fontsize=16)
        # ax.legend(loc='upper left', fontsize='large', ncol=1, bbox_to_anchor=(0.5, -0.7))
        plt.tight_layout()

        # Save each figure with a unique filename based on the score column
        if save_path:
            filename = save_path.replace(".png", f"_{column}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()

def plot_component_corrs(df, title=None, save_path=None):
    components = [
        'authenticity_score', 'empathy_score', 'engagement_score', 
        'emotion_provoking_score', 'narrative_complexity_score']
    
    labels = ["AUTH", "EMP", "ENG", "PROV", "NCOM", "HUM"]
    correlation_matrix = df[components].corr()

    # # Getting the Upper Triangle of the co-relation matrix
    # matrix = np.tril(correlation_matrix)

    # Create the heatmap with updated labels
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', cbar=True, square=True, # mask=matrix,
                        xticklabels=labels, yticklabels=labels, annot_kws={"size": 14}, linecolor='white', linewidths=1)

    # Title and labels
    plt.title(title, pad=20, fontsize=20)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def display_pairwise_t_tests(df, title=None, save_path=None):

    col = "model_short"
    components = [
        'authenticity_score', 'empathy_score', 'engagement_score', 
        'emotion_provoking_score', 'narrative_complexity_score']
    
    unique_vals = df[col].sort_values().unique()
    t_test_results = {}
    for component in components:
        for val1, val2 in itertools.combinations(unique_vals, 2):
            # Sort model names to ensure consistency
            model1, model2 = sorted([val1, val2])
            data_i = df[df[col] == model1][component]
            data_j = df[df[col] == model2][component]

            # Perform t-test between the two models
            t_stat, p_val = ttest_ind(data_i, data_j, nan_policy='omit')

            # Append the result
            t_test_results[(component,model1,model2)]= {
                'component': component,
                f'{col}_1': model1,
                f'{col}_2': model2,
                't_stat': t_stat,
                'p_value': p_val
            }

    df = pd.DataFrame.from_dict(t_test_results, orient='index')

    # Mapping for readable labels
    component_labels = {
        'authenticity_score': 'AUTH',
        'emotion_provoking_score': 'PROV',
        'empathy_score': 'EMP',
        'engagement_score': 'ENG',
        'narrative_complexity_score': 'NCOM'
    }
    
    # Creating a new column with combined model names
    df['model_comparison'] = df['model_short_1'] + " vs. " + df['model_short_2']

    # Creating pivot tables for t-statistics and p-values with combined model names
    pivot_table_t = df.pivot_table(index='model_comparison', columns='component', values='t_stat')
    pivot_table_p = df.pivot_table(index='model_comparison', columns='component', values='p_value')

    # Renaming the columns in the pivot tables
    pivot_table_t.rename(columns=component_labels, inplace=True)
    pivot_table_p.rename(columns=component_labels, inplace=True)

    # Plot heatmap for t-statistics (indicating direction and magnitude of change)
    plt.figure(figsize=(15, 10))
    
    # Create a custom annotator function to show p-values
    def annot_func(data):
        return data.map(lambda v: f"{v:.2f}")

    # Annotate heatmap with p-values
    annotations = annot_func(pivot_table_p)
    ax = sns.heatmap(
        pivot_table_t, 
        annot=annotations, 
        fmt="", 
        cmap='coolwarm_r', 
        center=0, 
        cbar_kws={'label': 't-statistic'}, 
        annot_kws={"size": 14},
        # vmin=-1, 
        # vmax=1
    )
    ax.figure.axes[-1].set_ylabel('T-Statistic', size=16)  

    # Title and labels
    plt.title(title, pad=20, fontsize=20)
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.xlabel('PSD Component', fontsize=18)
    plt.ylabel('Author Comparisons', fontsize=18)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":

    import os

    scores_path = "./human_study/data/processed/round1/human_annotations.csv"
    human_ratings_df = pd.read_csv(scores_path).drop(columns=["human_likeness_score"])

    sort_order = {
        "Llama-2-7B" :        0, 
        "Llama-2-13B":        1, 
        "Vicuna-33B" :        2,
        "Llama-2-70B":        3,
        "GPT-4":              4,
        "Human-Novice":       5,
        "Human-Intermediate": 6,
        "Human-Advanced":     7,
    }

    gpt4_vs_llms = list(sort_order)[:5]
    gpt4_vs_humans = list(sort_order)[4:]
    all_models = list(sort_order)

    llm_colors = plt.cm.Blues(np.linspace(0, 1, 7))[1:-1]
    human_colors = plt.cm.Reds(np.linspace(0, 1, 5))[1:-1]
    colors = np.concatenate((llm_colors, human_colors), axis=0)

    save_path = "./story_eval/imgs/"

    display_pairwise_t_tests(
        df=human_ratings_df,
        title='Authorship Comparisons',
        save_path=os.path.join(save_path, "pairwise_t-test_p-values.png")
    )

    plot_component_corrs(
        df=human_ratings_df,
        title='',
        save_path=os.path.join(save_path, "component_corrs.png")
    )

    # # Spider Plots
    # plot_spider(
    #     df=human_ratings_df, 
    #     models=gpt4_vs_humans, 
    #     title="", # Mean Psychological Depth Scores
    #     save_path=os.path.join(save_path, "mean_scores_gpt4_vs_humans.png"),
    #     sort_order=sort_order,
    #     colors=colors
    # )
    # plot_spider(
    #     df=human_ratings_df, 
    #     models=gpt4_vs_llms, 
    #     title="", # Mean Psychological Depth Scores
    #     save_path=os.path.join(save_path, "mean_scores_gpt4_vs_llms.png"),
    #     sort_order=sort_order,
    #     colors=colors
    # )
    # plot_spider(
    #     df=human_ratings_df, 
    #     models=all_models, 
    #     title="", # Mean Psychological Depth Scores
    #     save_path=os.path.join(save_path, "mean_scores_all.png"),
    #     sort_order=sort_order,
    #     colors=colors
    # )

    # CDF plots
    plot_cdf(
        df=human_ratings_df, 
        models=all_models, 
        title="Psychological Depth CDF", 
        save_path=os.path.join(save_path, "cdf_all_rough.png"),
        sort_order=sort_order,
        colors=colors,
        smooth=False
    )    
    plot_cdf(
        df=human_ratings_df, 
        models=all_models, 
        title="Psychological Depth CDF", 
        save_path=os.path.join(save_path, "cdf_all_smooth.png"),
        sort_order=sort_order,
        colors=colors,
        smooth=True
    )
    plot_cdf(
        df=human_ratings_df, 
        models=gpt4_vs_llms, 
        title="Psychological Depth CDF", 
        save_path=os.path.join(save_path, "cdf_gpt4_vs_llms_smooth.png"),
        sort_order=sort_order,
        colors=colors[:6],
        smooth=True
    )
    plot_cdf(
        df=human_ratings_df, 
        models=gpt4_vs_humans, 
        title="Psychological Depth CDF", 
        save_path=os.path.join(save_path, "cdf_gpt4_vs_humans_smooth.png"),
        sort_order=sort_order,
        colors=colors[4:],
        smooth=True
    )