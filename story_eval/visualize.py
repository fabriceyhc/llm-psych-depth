import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_spider(df, models, title, save_path=None):
  
    sort_order = {
        "Llama-2-7B" : 0, 
        "Llama-2-13B": 1, 
        "Vicuna-33B" : 2,
        "Llama-2-70B": 3,
        "GPT-4":       4,
        "Human-Low":   5,
        "Human-Mid":   6,
        "Human-High":  7,
    }

    # Filter data for specific models
    filtered_data = df[df["model_short"].isin(models)]

    # Remove 'participant_id' and 'story_id' from the aggregation
    filtered_data_no_ids = filtered_data.drop(columns=['participant_id', 'story_id', 'model_full', 'strategy'])

    # Re-aggregate the data without participant and story IDs
    aggregated_filtered_data_no_ids = filtered_data_no_ids.groupby('model_short').agg(['mean', 'var'])
    aggregated_filtered_data_no_ids.columns = ['_'.join(col).strip() for col in aggregated_filtered_data_no_ids.columns.values]
    aggregated_filtered_data_no_ids = aggregated_filtered_data_no_ids.sort_values(by=["model_short"], key=lambda x: x.map(sort_order))

    # Prepare data and labels for the radar chart
    labels = [label.split('_')[0].replace('_', ' ').title() for label in aggregated_filtered_data_no_ids.columns[::2]]
    labels = [label.replace('Human', 'Human Likeness').replace('Narrative', 'Narrative Complexity').replace('Emotion', 'Emotion Provoking') for label in labels]
    colors = plt.cm.Spectral_r(np.linspace(0, 1, len(sort_order)))

    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Add models to the radar chart
    for idx, (model_name, row) in enumerate(aggregated_filtered_data_no_ids.iterrows()):
        color = colors[sort_order[model_name]]
        add_radar_chart_filled_variance(ax, angles, row, model_name, color, variance_fraction=0.1)

    # Format the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=14)
    ax.set_yticks(np.arange(1, 6))
    ax.set_yticklabels(np.arange(1, 6), color="grey", size=12)
    plt.title(title, size=24, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()




if __name__ == "__main__":

    import os

    scores_path = "./human_study/data/processed/human_annotations.csv"
    human_ratings_df = pd.read_csv(scores_path)

    sort_order = {
        "Llama-2-7B" : 0, 
        "Llama-2-13B": 1, 
        "Vicuna-33B" : 2,
        "Llama-2-70B": 3,
        "GPT-4":       4,
        "Human-Low":   5,
        "Human-Mid":   6,
        "Human-High":  7,
    }

    save_path = "./story_eval/imgs/"

    plot_spider(
        df=human_ratings_df, 
        models=list(sort_order)[4:], 
        title="GPT-4 vs Human Quaility Levels", 
        save_path=os.path.join(save_path, "mean_scores_gpt4_vs_humans.png")
    )
    plot_spider(
        df=human_ratings_df, 
        models=list(sort_order)[:5], 
        title="GPT-4 vs Other LLMs", 
        save_path=os.path.join(save_path, "mean_scores_gpt4_vs_llms.png")
    )
    plot_spider(
        df=human_ratings_df, 
        models=list(sort_order), 
        title="All Models", 
        save_path=os.path.join(save_path, "mean_scores_all.png")
    )