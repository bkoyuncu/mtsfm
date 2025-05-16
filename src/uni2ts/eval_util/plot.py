from typing import Iterator, Optional

import matplotlib.pyplot as plt
# import 
import numpy as np
import pandas as pd
from gluonts import maybe
from gluonts.model import Forecast



def plot_single(
    inp: dict,
    label: dict,
    forecast: Forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.axis] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    ax = maybe.unwrap_or_else(ax, plt.gca)

    target = np.concatenate([inp["target"], label["target"]], axis=-1)
    start = inp["start"]
    if dim is not None:
        target = target[dim]
        forecast = forecast.copy_dim(dim)

    index = pd.period_range(start, periods=len(target), freq=start.freq)
    ax.plot(
        index.to_timestamp()[-context_length - forecast.prediction_length :],
        target[-context_length - forecast.prediction_length :],
        label="target",
        color="black",
    )
    forecast.plot(
        intervals=intervals,
        ax=ax,
        color="blue",
        name=name,
        show_label=show_label,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")

def plot_next_multi(
    axes: np.ndarray,
    input_it: Iterator[dict],
    label_it: Iterator[dict],
    forecast_it: Iterator[Forecast],
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
        plot_single(
            inp,
            label,
            forecast,
            context_length,
            intervals=intervals,
            ax=ax,
            dim=dim,
            name=name,
            show_label=show_label,
        )




def plot_single_ours(
    inp: dict,
    label: dict,
    forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.Axes] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
    plot_forecast: bool = False,
    column_names: list = None,
    plot_covariate: bool = False
):
    ax = ax if ax is not None else plt.gca()

    # Separate input and label targets
    input_target = inp["target"]
    label_target = label["target"]
    start = inp["start"]
    if dim is not None:
        input_target = input_target[dim]
        label_target = label_target[dim]
        forecast = forecast.copy_dim(dim)

    # Generate index for input and label
    input_index = pd.period_range(start, periods=len(input_target), freq=start.freq)
    label_index = pd.period_range(start + len(input_target), periods=len(label_target), freq=start.freq)

    # Plot input target
    ax.plot(
        input_index.to_timestamp()[-context_length+1:],
        input_target[-context_length+1:],
        label="input",
        color="black",
    )
    
    if plot_covariate:
        input_var = inp["past_feat_dynamic_real"][0]
        # Plot input target
        ax.plot(
                input_index.to_timestamp()[-context_length+1:],
                input_var[-context_length+1:],
                label="variate",
                color="brown",
            )


    # Plot label target
    ax.plot(
        label_index.to_timestamp()[:forecast.prediction_length],
        label_target[:forecast.prediction_length],
        label="label",
        color="red",
    )

    if plot_forecast:
        # Plot forecast
        forecast.plot(
            intervals=intervals,
            ax=ax,
            color="blue",
            name=name,
            show_label=show_label,
        )
    
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")
    try:
        ax.set_title(f"Dimension {dim}, Feature {column_names[dim]}" if dim is not None else "Forecast Plot")
    except: 
        ax.set_title(f"Dimension {dim}" if dim is not None else "Forecast Plot")



def plot_input(
    inp: dict,
    label: dict,
    forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.Axes] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
    plot_forecast: bool = False,
    column_names: list = None,
    plot_covariate: bool = False,
    label_str= None,
    title = None,
    color = None,
):
    ax = ax if ax is not None else plt.gca()

    input_target = inp["target"]
    label_target = label["target"]
    start = inp["start"]

    if plot_covariate:
        input_var = inp["past_feat_dynamic_real"][0]
        
        # Generate index for input and label
        input_index = pd.period_range(start, periods=len(input_target), freq=start.freq)
        label_index = pd.period_range(start + len(input_target), periods=len(label_target), freq=start.freq)

        # Plot input target
        ax.plot(
                input_index.to_timestamp()[-context_length+1:],
                input_var[-context_length+1:],
                label= label_str if label_str else 'Variate',
                color=color if color is not None else 'Brown',
            )

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")
    try:
        ax.set_title(f"Dimension {dim}, Feature {column_names[dim]}" if dim is not None else "Forecast Plot")
    except: 
        ax.set_title(f"Dimension {dim}" if dim is not None else "Forecast Plot")
    if title:
        ax.set_title(title)


def plot_single_ours_masked(
    inp: dict,
    label: dict,
    forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.Axes] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
    plot_forecast: bool = False,
    column_names: list = None,
    jump_mask=None,
    use_mean=True
):
    ax = ax if ax is not None else plt.gca()

    # Separate input and label targets
    input_target = inp["target"]
    label_target = label["target"]
    start = inp["start"]
    if dim is not None:
        input_target = input_target[dim]
        label_target = label_target[dim]
        jump_mask = jump_mask[dim]
        forecast = forecast.copy_dim(dim)

    # Generate index for input and label
    input_index = pd.period_range(start, periods=len(input_target), freq=start.freq)
    label_index = pd.period_range(start + len(input_target), periods=len(label_target), freq=start.freq)

    # Plot input target
    ax.plot(
        input_index.to_timestamp()[-context_length+1:],
        input_target[-context_length+1:],
        label="input",
        color="black",
    )
    
    # Plot label target
    ax.plot(
        label_index.to_timestamp()[:forecast.prediction_length],
        label_target[:forecast.prediction_length],
        label="label",
        color="red",
    )



    if jump_mask is not None:
        color = 'blue'
        name = 'prediction'

        if not isinstance(jump_mask, list):
            jump_mask =  [jump_mask]

        frc_samples = forecast.samples.copy()
        samples_t = frc_samples.transpose(1,0)
        for mask in jump_mask:

            samples = samples_t[mask] #check shapes (L, #Samples)
            pseudo_len = samples.shape[0]
            
            jump_date = np.argmax(mask) if np.any(mask) else mask.shape[0]
            # get jump date intervals
            date_plot =forecast.start_date+jump_date
            dates_to_print = pd.period_range(date_plot, periods=pseudo_len, freq=forecast.start_date.freq)

            #plot vertical line
            # ax.axvline(x=date_plot.to_timestamp(), color='k', linestyle='--', label='Jump Date')
            
            #get mean over L
            if use_mean:
                samples_mean = np.mean(samples, axis = 0)
                median = np.median(samples_mean, axis=0)
            else:
                samples_mean = samples
                median = np.median(samples_mean)
            # bp = ax.boxplot([samples_mean], positions=[date_plot.to_timestamp()], patch_artist=True)
            #get median of the samples
            
            
            # Plot median forecast
            ax.plot(dates_to_print.to_timestamp(), [median] *pseudo_len, color=color, label=name)
            # print('prediction', median, date_plot.to_timestamp())

            # Plot prediction intervals
            for interval in intervals:
                if show_label:
                    if name is not None:
                        label = f"{name}: {interval}"
                    else:
                        label = interval
                else:
                    label = None

                low = (1 - interval) / 2
                high = 1 - low
                print(   [np.quantile(samples_mean, low)],    [np.quantile(samples_mean, high)])

                ax.fill_between(
                    dates_to_print.to_timestamp(),
                    [np.quantile(samples_mean, low)]*pseudo_len,
                    [np.quantile(samples_mean, high)]*pseudo_len,
                    alpha=0.5 - interval / 3,
                    facecolor=color,
                label=label,
            )

     
    # if plot_forecast:
    #     # Plot forecast
    #     forecast.plot(
    #         intervals=intervals,
    #         ax=ax,
    #         color="blue",
    #         name=name,
    #         show_label=show_label,
    #     )
    
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")

    handles = labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    try:
        ax.set_title(f"Dimension {dim}, Feature {column_names[dim]}" if dim is not None else "Forecast Plot")
    except: 
        ax.set_title(f"Dimension {dim}" if dim is not None else "Forecast Plot")