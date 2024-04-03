import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


ENV_COLORS = {
    "Baseline": "y",
    "Baseline-v0": "y",
    "Enhanced": "b",
    "Enhanced-v0": "b",
    "Enhanced-v1": "g",
    "Enhanced-v0 - CAPS": "m",
    "Enhanced-v1 - CAPS": "c",
}


def calculate_ema(data, alpha):
    ema = np.zeros_like(data)  # Initialize an array for the EMA values
    ema[0] = data[0]  # Set the initial value of the EMA to the first data point

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]  # Calculate the EMA

    return ema


def get_table(df_crude, environments, var):
    columns = ["Step"]
    for env in environments:
        columns.append(f"{env} - ep_info/{var}")
    return df_crude[columns].dropna()


def plot_steps(
    steps_table,
    environments,
    n_steps=150000,
    max_steps=1200,
    max_time=30,
):
    for environment in environments:
        steps_table[f"{environment}-Smoothed"] = calculate_ema(
            steps_table[f"{environment} - ep_info/steps"].values, 1 - 0.999
        )
        steps_table[f"{environment}-Smoothed"] = calculate_ema(
            steps_table[f"{environment} - ep_info/steps"].values, 1 - 0.999
        )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ps = []
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    for environment in environments:
        x = steps_table["Step"][steps_table["Step"] <= n_steps]
        y = steps_table[f"{environment}-Smoothed"][steps_table["Step"] <= n_steps]
        y2 = y * 0.025
        # Plot x vs y and x vs y2 in a twin plot
        (p,) = ax1.plot(x, y, f"{ENV_COLORS[environment]}-", label=f"{environment}")
        ps.append(p)

        if environment == environments[0]:
            ax2 = ax1.twinx()
            ax2.plot(x, y2, f"{ENV_COLORS[environment]}-")
            ax2.set_ylabel("Seconds", color="r")
            ax2.tick_params("y", colors="r")
            ax2.set_ylim([0, max_time])
    ax1.set_xlabel("Training Steps")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    ax1.set_ylabel("Steps", color=f"b")
    ax1.tick_params("y", colors=f"b")
    ax1.set_ylim([0, max_steps])
    fig.tight_layout()
    plt.title("Episodic Length")
    fig.gca().xaxis.set_major_formatter(formatter)
    plt.show()


def plot_cpad(
    cpad_table,
    environments,
    n_steps=150000,
    log_scale=True,
    y_ticks=[1, 10, 100],
):
    for environment in environments:
        cpad_table[f"{environment}-Smoothed"] = calculate_ema(
            cpad_table[f"{environment} - ep_info/action_var"].values, 1 - 0.999
        )
        cpad_table[f"{environment}-Smoothed"] = calculate_ema(
            cpad_table[f"{environment} - ep_info/action_var"].values, 1 - 0.999
        )
    fig, ax = plt.subplots(figsize=(10, 5))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    for environement in environments:
        x = cpad_table["Step"][cpad_table["Step"] <= n_steps]
        y = cpad_table[f"{environement}-Smoothed"][cpad_table["Step"] <= n_steps]
        ax.plot(x, y, f"{ENV_COLORS[environement]}-", label=f"{environement}")
    ax.grid(True)
    ax.legend(loc="upper right")
    y_label = "CPAD"
    if log_scale:
        plt.yscale("log")
        y_label += " (log scale)"
    plt.yticks(y_ticks)
    plt.xlabel("Training Steps")
    plt.ylabel(y_label)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.title("Cumulative Pairwise Action Distance")
    plt.show()


def box_plot_steps(data_frames, colors, name_to_save="0"):
    fig, ax = plt.subplots()
    plt.grid(True)
    ax2 = ax.twinx()
    ax.set_ylabel("Steps", weight="bold", color="b")
    ax2.set_ylabel("Seconds", weight="bold", color="r")
    # ax.set_xlabel("Environment")
    ax.set_title("Episode Length", weight="bold", fontsize=16)
    ax.set_ylim(0, 1300)
    ax2.set_ylim(0, 1300 * 0.025)
    for df in data_frames:
        sns.boxplot(
            data=df,
            x="Environment",
            y="Episode Length",
            ax=ax,
            color=colors[df["Environment"].iloc[0]],
        )
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("")
    plt.savefig(f"epi_length_{name_to_save}.png")
    plt.show()


def box_plot_cpad(data_frames, colors, name_to_save="0"):
    fig, ax = plt.subplots()
    ax.set_ylabel("CPAD", weight="bold")
    # ax.set_xlabel("Environment")
    ax.set_title("Cumulative Pairwise Action Distance", weight="bold", fontsize=16)
    for df in data_frames:
        sns.boxplot(
            data=df,
            x="Environment",
            y="Cumulative Pairwise Action Distance",
            ax=ax,
            color=colors[df["Environment"].iloc[0]],
        )
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("")
    plt.grid(True)
    plt.savefig(f"cpad_{name_to_save}.png")
    plt.show()
