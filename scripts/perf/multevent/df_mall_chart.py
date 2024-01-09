#!/usr/bin/env python3

import sys
import pandas as pd
import plotext as plt
import argparse

# from tabulate import tabulate #debug only


def plot_mall_chart(df):
    """
    Plot key metrics of MALL for MI300 GPU from given dataframe.
    """

    # NB: 
    #  - The fields and groups shold be designed to be easily
    #    customized for other fields, or passed in as parameters.
    #  - Potentially, we could list all available fields if need it.

    # The interested field and its attributes
    fields = {
        "Total Hit Rate": {"unit": "%", "label": "Total"},
        "Total Read Hit Rate": {"unit": "%", "label": "Read"},
        "Total Write Hit Rate": {"unit": "%", "label": "Write"},
        "Total Read bandwidth": {"unit": "GB/s", "label": "Read"},
        "Total Write bandwidth": {"unit": "GB/s", "label": "Write"},
    }

    # Group the fields to show details per bank.
    # NB: dataclasses may be better.
    groups = [
        {
            "title": "Hit Rate Details (%)",
            "items": ["Total Hit Rate", "Total Read Hit Rate", "Total Write Hit Rate"],
            "height": 16,
            "y_lim": 100,
        },
        {
            "title": "Bandwidth Details (GB/s)",
            "items": ["Total Read bandwidth", "Total Write bandwidth"],
            "height": 16,
            "y_lim": None,
        },
    ]

    ss = ""  # the return stringstream

    field_dfs = {}

    # print(df.columns)

    # remove white space before indexing and comparison
    new_labels = {}
    for c in df.columns:
        new_labels[c] = c.strip()
    df.rename(columns=new_labels, inplace=True)
    # apply strip() to the whole df?

    if len(df.columns) == 4:  # MI300A
        if (
            df.columns[0] != "IP"
            or df.columns[1] != "GPUDF_0"
            or df.columns[2] != "GPUDF_1"
            or df.columns[3] != "GPUDF_2"
        ):
            sys.exit(
                "Header doesn't match on MI300A! There might some changes from multevent csv."
            )
    elif len(df.columns) == 5:  # MI300X, not tested yet
        # print(df.columns)
        if (
            df.columns[0] != "IP"
            or df.columns[1] != "GPUDF_0"
            or df.columns[2] != "GPUDF_1"
            or df.columns[3] != "GPUDF_2"
            or df.columns[4] != "GPUDF_3"
        ):
            sys.exit(
                "Header doesn't match  on MI300X! There might some changes from multevent csv."
            )
    else:
        sys.exit("Header doesn't match! There might some changes from multevent csv.")

    # print(df["IP"])

    # Aggregated summary
    ss += "\n" + "-" * 160 + "\nMALL Speed-of-Light\n"

    for k in fields.keys():
        field_dfs[k] = df.loc[df["IP"].str.contains(k)]  # df.query() ?
        field_dfs[k].reset_index(drop=True, inplace=True)
        del field_dfs[k]["IP"]  # remove non-numerical column
        field_dfs[k] = field_dfs[k].astype(float)
        # print(tabulate(field_dfs[k], headers="keys", tablefmt="fancy_grid"))
        ss += "    Averaged {title:<22} {val:>6.2f} {unit} ".format(
            title=k.title(), val=field_dfs[k].mean().mean(), unit=fields[k]["unit"]
        )
        ss += "\n"

    ss += "\n" + "-" * 160 + "\n"

    # Details in group
    for it in groups:
        ss += it["title"] + "\n"

        for i in range(0, len(df.columns) - 1):
            # NB: think about subplot for overall html output
            plt.clear_figure()
            plt.theme("pro")
            data = []
            labels = []
            for item in it["items"]:
                data.append(field_dfs[item]["GPUDF_" + str(i)].tolist())
                labels.append(fields[item]["label"])

            plt.multiple_bar(
                field_dfs[it["items"][0]].index.values.tolist(),
                data,
                label=labels,
                color=["blue", "blue+", 68, 63],
            )
            sub_title = "AID"
            sub_title += str(i + 1) if len(df.columns) == 4 else str(i)
            sub_title += "(with XCD" + str(i * 2)
            sub_title += " XCD" + str(i * 2 + 1) + ")"
            plt.title(sub_title)
            plt.plot_size(height=it["height"])
            if it["y_lim"] != None:
                plt.ylim(upper=it["y_lim"])

            ss += plt.build() + "\n"

            # Todo: maybe all in plotext
            # plt.save_fig("./debug.html", keep_colors = True, append = True)

        ss += "\n" + "-" * 160 + "\n"

    return ss


if __name__ == "__main__":
    """
    Plot key metrics of MALL for MI300 GPU with given csv generated from multevent.
    """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--start_line',
                        default=18,
                        type=int,
                        help='Specify the line start to read.')

    parser.add_argument("df_file", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    df = pd.read_csv(
        args.df_file[0],
        header=args.start_line,  # pick up specific line holding AID# as the header
        # on_bad_lines='warn',
        skipinitialspace=True,
    ).dropna()
    print(plot_mall_chart(df))
