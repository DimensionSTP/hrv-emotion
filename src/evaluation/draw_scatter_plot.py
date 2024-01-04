
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from matplotlib import pyplot as plt
from matplotlib import font_manager, rc

def draw_scatter_plot(x, y, scale):
    df = pd.read_excel("survey.xlsx")

    df["lnHF/lnLF"] = np.nan
    df["lnHF/lnLF"] = df["lnHF"] / df["lnLF"]
    
    df["ln(HF/LF)"] = np.nan
    df["ln(HF/LF)"] = np.log(df["HF"] / df["LF"])

    df["stimuli_arousal"] = np.nan
    df.loc[df["Arousal"] >= 5, "stimuli_arousal"] = "각성"
    df.loc[df["Arousal"] == 4, "stimuli_arousal"] = "중립"
    df.loc[df["Arousal"] <= 3, "stimuli_arousal"] = "이완"
    
    df_arousal = df[df["stimuli_arousal"] == "각성"]
    df_neutral = df[df["stimuli_arousal"] == "중립"]
    df_relax = df[df["stimuli_arousal"] == "이완"]

    x_arousal = df_arousal[x].values
    y_arousal = df_arousal[y].values
    x_arousal = x_arousal.reshape(-1, 1)
    y_arousal = y_arousal.reshape(-1, 1)

    x_neutral = df_neutral[x].values
    y_neutral = df_neutral[y].values
    x_neutral = x_neutral.reshape(-1, 1)
    y_neutral = y_neutral.reshape(-1, 1)

    x_relax = df_relax[x].values
    y_relax = df_relax[y].values
    x_relax = x_relax.reshape(-1, 1)
    y_relax = y_relax.reshape(-1, 1)

    lr_arousal = LinearRegression()
    lr_neutral = LinearRegression()
    lr_relax = LinearRegression()

    lr_arousal.fit(x_arousal, y_arousal)
    lr_neutral.fit(x_neutral, y_neutral)
    lr_relax.fit(x_relax, y_relax)
    
    x_arousal_range = np.array([min(x_arousal), max(x_arousal)])
    y_arousal_fit = x_arousal_range * lr_arousal.coef_[0] + lr_arousal.intercept_
    x_neutral_range = np.array([min(x_neutral), max(x_neutral)])
    y_neutral_fit = x_neutral_range * lr_neutral.coef_[0] + lr_neutral.intercept_
    x_relax_range = np.array([min(x_relax), max(x_relax)])
    y_relax_fit = x_relax_range * lr_relax.coef_[0] + lr_relax.intercept_

    results_arousal = sm.OLS(y_arousal, sm.add_constant(x_arousal)).fit()
    results_neutral = sm.OLS(y_neutral, sm.add_constant(x_neutral)).fit()
    results_relax = sm.OLS(y_relax, sm.add_constant(x_relax)).fit()

    groups = df.groupby("stimuli_arousal")

    font_path = "C:/Windows/Fonts/NGULIM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    fig, ax = plt.subplots()

    for name, group in groups:
        ax.plot(
            group[x], 
            group[y], 
            marker = ".",
            linestyle = "",
            label = name
            )
        
    ax.plot(x_arousal_range, y_arousal_fit, color = "blue", label = "각성")
    ax.plot(x_relax_range, y_relax_fit, color = "orange", label = "이완")
    ax.plot(x_neutral_range, y_neutral_fit, color = "green", label = "중립")
    
    ax.legend(fontsize=8, loc="upper left")
    plt.axis("square")
    plt.title(f"{x}, {y} 산포도", fontsize=16)
    plt.xlabel(f"{x}", fontsize=10)
    plt.ylabel(f"{y}", fontsize=10)
    plt.xticks(scale)
    plt.yticks(scale)
    plt.show()
    
    print(results_arousal.summary())
    print(results_neutral.summary())
    print(results_relax.summary())
    
if __name__ == "__main__":
    # draw_scatter_plot("HF", "LF", np.arange(0, 2500, 250))
    # draw_scatter_plot("lnLF", "lnHF", np.arange(0, 11, 1))
    # draw_scatter_plot("HFp", "LFp", np.arange(0, 0.9, 0.1))
    # draw_scatter_plot("HF", "VLF", np.arange(0, 2500, 250))
    # draw_scatter_plot("lnHF", "lnVLF", np.arange(0, 11, 1))
    # draw_scatter_plot("HFp", "VLFp", np.arange(0, 0.9, 0.1))
    # draw_scatter_plot("SDNN", "rMSSD", np.arange(0, 90, 10))
    # draw_scatter_plot("tPow", "pPow", np.arange(0, 4500, 500))
    # draw_scatter_plot("lnLF", "lnHF/lnLF", np.arange(0, 8, 0.5))
    # draw_scatter_plot("lnHF", "lnHF/lnLF", np.arange(0, 8, 0.5))
    draw_scatter_plot("lnHF", "ln(HF/LF)", np.arange(-3, 8, 1))
    draw_scatter_plot("lnLF", "ln(HF/LF)", np.arange(-3, 8, 1))