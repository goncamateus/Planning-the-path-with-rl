import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    environements,
    n_steps=150000,
    max_steps=1200,
    max_time=30,
):
    for environment in environements:
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
    for environment in environements:
        x = steps_table["Step"][steps_table["Step"] <= n_steps]
        y = steps_table[f"{environment}-Smoothed"][steps_table["Step"] <= n_steps]
        y2 = y * 0.025
        # Plot x vs y and x vs y2 in a twin plot
        (p,) = ax1.plot(x, y, f"{ENV_COLORS[environment]}-", label=f"{environment}")
        ps.append(p)

        if environment == environements[0]:
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
    environements,
    n_steps=150000,
    y_ticks=[1, 10, 100],
):
    for environment in environements:
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
    for environement in environements:
        x = cpad_table["Step"][cpad_table["Step"] <= n_steps]
        y = cpad_table[f"{environement}-Smoothed"][cpad_table["Step"] <= n_steps]
        ax.plot(x, y, f"{ENV_COLORS[environement]}-", label=f"{environement}")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.yticks(y_ticks)
    plt.yscale("log")
    plt.xlabel("Training Steps")
    plt.ylabel("CPAD (log scale)")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.title("Cumulative Pairwise Action Distance")
    plt.show()
