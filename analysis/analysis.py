import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")

# Define the header columns
header = [
    "name",
    "model",
    "id",
    "task",
    "var",
    "type",
    "num_steps",
    "success",
    "points",
    "golden",
]


def load_data():
    """Load all CSV files from the data directory"""
    data_dir = Path("analysis/data")
    csv_files = glob.glob(str(data_dir / "*.csv"))

    all_data = []

    for file_path in csv_files:
        file_name = Path(file_path).stem

        # Skip empty files
        if Path(file_path).stat().st_size == 0:
            print(f"Skipping empty file: {file_name}")
            continue

        try:
            df = pd.read_csv(file_path, header=None, names=header)
            df["source_file"] = file_name
            all_data.append(df)
            print(f"Loaded {len(df)} records from {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        print("No data loaded!")
        return pd.DataFrame()


def extract_model_info(df):
    """Extract model information from the data"""
    # Extract model names from the model column
    df["model_short"] = df["model"].str.split("/").str[-1]

    # Extract version info from name column
    df["version"] = df["name"].str.extract(r"-(v\d+)-")[0]

    return df


def analyze_success_rates(df):
    """Analyze success rates by model and task"""
    print("\n=== SUCCESS RATE ANALYSIS ===")

    # Overall success rates by model
    success_by_model = (
        df.groupby("model_short")["success"].agg(["count", "sum", "mean"]).round(3)
    )
    success_by_model.columns = ["total_tasks", "successful_tasks", "success_rate"]
    print("\nSuccess rates by model:")
    print(success_by_model.sort_values("success_rate", ascending=False))

    # Success rates by task type
    success_by_task = df.groupby("task")["success"].agg(["count", "sum", "mean"]).round(3)
    success_by_task.columns = ["total_attempts", "successful_attempts", "success_rate"]
    print("\nSuccess rates by task (top 10 most attempted):")
    print(success_by_task.sort_values("total_attempts", ascending=False).head(10))

    return success_by_model, success_by_task


def analyze_performance_metrics(df):
    """Analyze performance metrics like steps and points"""
    print("\n=== PERFORMANCE METRICS ANALYSIS ===")

    # Average number of steps by model
    steps_by_model = (
        df.groupby("model_short")["num_steps"].agg(["mean", "median", "std"]).round(2)
    )
    print("\nSteps taken by model:")
    print(steps_by_model.sort_values("mean"))

    # Average points by model (for successful tasks only)
    successful_df = df[df["success"] == True]
    points_by_model = (
        successful_df.groupby("model_short")["points"]
        .agg(["mean", "median", "std"])
        .round(2)
    )
    print("\nPoints earned by model (successful tasks only):")
    print(points_by_model.sort_values("mean", ascending=False))

    # Efficiency: points per step for successful tasks
    successful_df["efficiency"] = successful_df["points"] / successful_df["num_steps"]
    efficiency_by_model = (
        successful_df.groupby("model_short")["efficiency"]
        .agg(["mean", "median"])
        .round(3)
    )
    print("\nEfficiency (points per step) by model:")
    print(efficiency_by_model.sort_values("mean", ascending=False))

    return steps_by_model, points_by_model, efficiency_by_model


def analyze_task_difficulty(df):
    """Analyze task difficulty based on success rates and steps"""
    print("\n=== TASK DIFFICULTY ANALYSIS ===")

    task_stats = (
        df.groupby("task")
        .agg({
            "success": ["count", "mean"],
            "num_steps": "mean",
            "points": "mean",
            "golden": "mean",
        })
        .round(2)
    )

    task_stats.columns = [
        "attempts",
        "success_rate",
        "avg_steps",
        "avg_points",
        "avg_golden",
    ]
    task_stats["difficulty_score"] = (1 - task_stats["success_rate"]) * task_stats[
        "avg_steps"
    ]

    print("\nTask difficulty ranking (top 10 hardest):")
    print(task_stats.sort_values("difficulty_score", ascending=False).head(10))

    print("\nEasiest tasks (top 10):")
    print(task_stats.sort_values("difficulty_score").head(10))

    return task_stats


def create_visualizations(df, success_by_model, task_stats):
    """Create visualizations of the analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Success rates by model
    success_by_model["success_rate"].plot(kind="bar", ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Success Rate by Model")
    axes[0, 0].set_ylabel("Success Rate")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Distribution of steps taken
    df.boxplot(column="num_steps", by="model_short", ax=axes[0, 1])
    axes[0, 1].set_title("Distribution of Steps by Model")
    axes[0, 1].set_ylabel("Number of Steps")
    plt.suptitle("")  # Remove default title

    # 3. Points distribution for successful tasks
    successful_df = df[df["success"] == True]
    if not successful_df.empty:
        successful_df.boxplot(column="points", by="model_short", ax=axes[1, 0])
        axes[1, 0].set_title("Points Distribution by Model (Successful Tasks)")
        axes[1, 0].set_ylabel("Points")
        plt.suptitle("")

    # 4. Task difficulty heatmap
    top_tasks = task_stats.head(15)  # Top 15 tasks
    task_matrix = df[df["task"].isin(top_tasks.index)].pivot_table(
        values="success", index="task", columns="model_short", aggfunc="mean"
    )
    sns.heatmap(
        task_matrix,
        annot=True,
        cmap="RdYlGn",
        ax=axes[1, 1],
        cbar_kws={"label": "Success Rate"},
    )
    axes[1, 1].set_title("Success Rate Heatmap by Task and Model")

    plt.tight_layout()
    plt.savefig("analysis_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def compare_versions(df):
    """Compare different versions (v1 vs v2) if available"""
    print("\n=== VERSION COMPARISON ===")

    if "version" in df.columns and df["version"].notna().any():
        version_stats = (
            df.groupby(["model_short", "version"])
            .agg({"success": ["count", "mean"], "num_steps": "mean", "points": "mean"})
            .round(3)
        )

        print("Performance by model version:")
        print(version_stats)
    else:
        print("No version information found in the data.")


def main():
    """Main analysis function"""
    print("Loading and analyzing model performance data...")

    # Load data
    df = load_data()
    if df.empty:
        return

    # Extract model information
    df = extract_model_info(df)

    print(f"\nTotal records loaded: {len(df)}")
    print(f"Models: {df['model_short'].unique()}")
    print(f"Tasks: {len(df['task'].unique())} unique tasks")
    print(f"Source files: {df['source_file'].unique()}")

    # Perform analyses
    success_by_model, success_by_task = analyze_success_rates(df)
    steps_by_model, points_by_model, efficiency_by_model = analyze_performance_metrics(df)
    task_stats = analyze_task_difficulty(df)
    compare_versions(df)

    # Create visualizations
    create_visualizations(df, success_by_model, task_stats)

    # Summary insights
    print("\n=== KEY INSIGHTS ===")
    best_model = success_by_model["success_rate"].idxmax()
    worst_model = success_by_model["success_rate"].idxmin()

    print(
        f"🏆 Best performing model: {best_model} ({success_by_model.loc[best_model, 'success_rate']:.1%} success rate)"
    )
    print(
        f"📉 Lowest performing model: {worst_model} ({success_by_model.loc[worst_model, 'success_rate']:.1%} success rate)"
    )

    hardest_task = task_stats["difficulty_score"].idxmax()
    easiest_task = task_stats["difficulty_score"].idxmin()

    print(
        f"🔥 Hardest task: {hardest_task} (difficulty score: {task_stats.loc[hardest_task, 'difficulty_score']:.2f})"
    )
    print(
        f"✅ Easiest task: {easiest_task} (difficulty score: {task_stats.loc[easiest_task, 'difficulty_score']:.2f})"
    )

    if not efficiency_by_model.empty:
        most_efficient = efficiency_by_model["mean"].idxmax()
        print(
            f"⚡ Most efficient model: {most_efficient} ({efficiency_by_model.loc[most_efficient, 'mean']:.3f} points/step)"
        )


if __name__ == "__main__":
    main()
