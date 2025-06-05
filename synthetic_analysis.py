import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_synthetic_results():

    file_path = os.path.dirname(__file__)
    results_dir = os.path.join(file_path, "results")

    for data_to_view in ["synthetic-unet", "synthetic-unetr", "synthetic-atunet"]:
        data_96_1 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-96-1.csv")
        )
        data_96_1["Image_size"] = [96] * 100
        data_96_2 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-96-2.csv")
        )
        data_96_2["Image_size"] = [96] * 100
        data_96_3 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-96-3.csv")
        )
        data_96_3["Image_size"] = [96] * 100

        data_80_1 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-80-1.csv")
        )
        data_80_1["Image_size"] = [80] * 100
        data_80_2 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-80-2.csv")
        )
        data_80_2["Image_size"] = [80] * 100
        data_80_3 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-80-3.csv")
        )
        data_80_3["Image_size"] = [80] * 100

        data_64_1 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-64-1.csv")
        )
        data_64_1["Image_size"] = [64] * 100
        data_64_2 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-64-2.csv")
        )
        data_64_2["Image_size"] = [64] * 100
        data_64_3 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-64-3.csv")
        )
        data_64_3["Image_size"] = [64] * 100

        data_48_1 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-48-1.csv")
        )
        data_48_1["Image_size"] = [48] * 100
        data_48_2 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-48-2.csv")
        )
        data_48_2["Image_size"] = [48] * 100
        data_48_3 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-48-3.csv")
        )
        data_48_3["Image_size"] = [48] * 100

        data_32_1 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-32-1.csv")
        )
        data_32_1["Image_size"] = [32] * 100
        data_32_2 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-32-2.csv")
        )
        data_32_2["Image_size"] = [32] * 100
        data_32_3 = pd.read_csv(
            os.path.join(results_dir, data_to_view, data_to_view + "-32-3.csv")
        )
        data_32_3["Image_size"] = [32] * 100

        combined_frames = [
            data_96_1,
            data_96_2,
            data_96_3,
            data_80_1,
            data_80_2,
            data_80_3,
            data_64_1,
            data_64_2,
            data_64_3,
            data_48_1,
            data_48_2,
            data_48_3,
            data_32_1,
            data_32_2,
            data_32_3,
        ]

        combined_df = pd.concat(combined_frames)
        combined_df.rename(columns={"Image_size": "PatchSize"}, inplace=True)

        combined_df["fg_group"] = pd.cut(
            combined_df["fg_ratio"],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
            labels=["0", "0.1", "0.2", "0.3", "0.4"],
            include_lowest=True,
        )

        visualize_df = combined_df.loc[
            (combined_df["PatchSize"] == 96)
            | (combined_df["PatchSize"] == 64)
            | (combined_df["PatchSize"] == 32)
        ]

        fg_ratio_32 = pd.read_csv(
            os.path.join(results_dir, data_to_view, "foreground_ratios-32.csv")
        )
        fg_ratio_48 = pd.read_csv(
            os.path.join(results_dir, data_to_view, "foreground_ratios-48.csv")
        )
        fg_ratio_64 = pd.read_csv(
            os.path.join(results_dir, data_to_view, "foreground_ratios-64.csv")
        )
        fg_ratio_80 = pd.read_csv(
            os.path.join(results_dir, data_to_view, "foreground_ratios-80.csv")
        )
        fg_ratio_96 = pd.read_csv(
            os.path.join(results_dir, data_to_view, "foreground_ratios-96.csv")
        )

        selected_dice_96 = []
        selected_dice_80 = []
        selected_dice_64 = []
        selected_dice_48 = []
        selected_dice_32 = []

        for index in range(len(combined_df)):
            row = combined_df.iloc[index]
            if row["PatchSize"] == 96:
                if (row["fg_ratio"] > fg_ratio_96.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_96.max()[0]
                ):
                    selected_dice_96.append(row["dsc"])
            elif row["PatchSize"] == 80:
                if (row["fg_ratio"] > fg_ratio_80.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_80.max()[0]
                ):
                    selected_dice_80.append(row["dsc"])
            elif row["PatchSize"] == 64:
                if (row["fg_ratio"] > fg_ratio_64.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_64.max()[0]
                ):
                    selected_dice_64.append(row["dsc"])
            elif row["PatchSize"] == 48:
                if (row["fg_ratio"] > fg_ratio_48.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_48.max()[0]
                ):
                    selected_dice_48.append(row["dsc"])
            elif row["PatchSize"] == 32:
                if (row["fg_ratio"] > fg_ratio_32.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_32.max()[0]
                ):
                    selected_dice_32.append(row["dsc"])

        mean_within_dice_96 = np.nanmean(selected_dice_96)
        mean_within_dice_80 = np.nanmean(selected_dice_80)
        mean_within_dice_64 = np.nanmean(selected_dice_64)
        mean_within_dice_48 = np.nanmean(selected_dice_48)
        mean_within_dice_32 = np.nanmean(selected_dice_32)

        std_within_dice_96 = np.nanstd(selected_dice_96)
        std_within_dice_80 = np.nanstd(selected_dice_80)
        std_within_dice_64 = np.nanstd(selected_dice_64)
        std_within_dice_48 = np.nanstd(selected_dice_48)
        std_within_dice_32 = np.nanstd(selected_dice_32)

        print("Dice 96 within:", mean_within_dice_96, " sd: (", std_within_dice_96, ")")
        print("Dice 80 within:", mean_within_dice_80, " sd: (", std_within_dice_80, ")")
        print("Dice 64 within:", mean_within_dice_64, " sd: (", std_within_dice_64, ")")
        print("Dice 48 within:", mean_within_dice_48, " sd: (", std_within_dice_48, ")")
        print("Dice 32 within:", mean_within_dice_32, " sd: (", std_within_dice_32, ")")

        selected_dice_96 = []
        selected_dice_80 = []
        selected_dice_64 = []
        selected_dice_48 = []
        selected_dice_32 = []

        for index in range(len(combined_df)):
            row = combined_df.iloc[index]
            if row["PatchSize"] == 96:
                if (row["fg_ratio"] <= fg_ratio_96.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_96.max()[0]
                ):
                    selected_dice_96.append(row["dsc"])
            elif row["PatchSize"] == 80:
                if (row["fg_ratio"] <= fg_ratio_80.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_80.max()[0]
                ):
                    selected_dice_80.append(row["dsc"])
            elif row["PatchSize"] == 64:
                if (row["fg_ratio"] <= fg_ratio_64.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_64.max()[0]
                ):
                    selected_dice_64.append(row["dsc"])
            elif row["PatchSize"] == 48:
                if (row["fg_ratio"] <= fg_ratio_48.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_48.max()[0]
                ):
                    selected_dice_48.append(row["dsc"])
            elif row["PatchSize"] == 32:
                if (row["fg_ratio"] <= fg_ratio_32.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_32.max()[0]
                ):
                    selected_dice_32.append(row["dsc"])

        mean_outside_dice_96 = np.nanmean(selected_dice_96)
        mean_outside_dice_80 = np.nanmean(selected_dice_80)
        mean_outside_dice_64 = np.nanmean(selected_dice_64)
        mean_outside_dice_48 = np.nanmean(selected_dice_48)
        mean_outside_dice_32 = np.nanmean(selected_dice_32)

        std_outside_dice_96 = np.nanstd(selected_dice_96)
        std_outside_dice_80 = np.nanstd(selected_dice_80)
        std_outside_dice_64 = np.nanstd(selected_dice_64)
        std_outside_dice_48 = np.nanstd(selected_dice_48)
        std_outside_dice_32 = np.nanstd(selected_dice_32)

        print(
            "Dice 96 outside drop:",
            mean_within_dice_96 - mean_outside_dice_96,
            " sd: (",
            std_outside_dice_96,
            ")",
        )
        print(
            "Dice 80 outside drop:",
            mean_within_dice_80 - mean_outside_dice_80,
            " sd: (",
            std_outside_dice_80,
            ")",
        )
        print(
            "Dice 64 outside drop:",
            mean_within_dice_64 - mean_outside_dice_64,
            " sd: (",
            std_outside_dice_64,
            ")",
        )
        print(
            "Dice 48 outside drop:",
            mean_within_dice_48 - mean_outside_dice_48,
            " sd: (",
            std_outside_dice_48,
            ")",
        )
        print(
            "Dice 32 outside drop:",
            mean_within_dice_32 - mean_outside_dice_32,
            " sd: (",
            std_outside_dice_32,
            ")",
        )

        selected_hd_96 = []
        selected_hd_80 = []
        selected_hd_64 = []
        selected_hd_48 = []
        selected_hd_32 = []

        for index in range(len(combined_df)):
            row = combined_df.iloc[index]
            if row["PatchSize"] == 96:
                if (row["fg_ratio"] > fg_ratio_96.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_96.max()[0]
                ):
                    selected_hd_96.append(row["hausdorff"])
            elif row["PatchSize"] == 80:
                if (row["fg_ratio"] > fg_ratio_80.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_80.max()[0]
                ):
                    selected_hd_80.append(row["hausdorff"])
            elif row["PatchSize"] == 64:
                if (row["fg_ratio"] > fg_ratio_64.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_64.max()[0]
                ):
                    selected_hd_64.append(row["hausdorff"])
            elif row["PatchSize"] == 48:
                if (row["fg_ratio"] > fg_ratio_48.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_48.max()[0]
                ):
                    selected_hd_48.append(row["hausdorff"])
            elif row["PatchSize"] == 32:
                if (row["fg_ratio"] > fg_ratio_32.min()[0]) and (
                    row["fg_ratio"] < fg_ratio_32.max()[0]
                ):
                    selected_hd_32.append(row["hausdorff"])

        mean_within_hd_96 = np.nanmean(selected_hd_96)
        mean_within_hd_80 = np.nanmean(selected_hd_80)
        mean_within_hd_64 = np.nanmean(selected_hd_64)
        mean_within_hd_48 = np.nanmean(selected_hd_48)
        mean_within_hd_32 = np.nanmean(selected_hd_32)

        std_within_hd_96 = np.nanstd(selected_hd_96)
        std_within_hd_80 = np.nanstd(selected_hd_80)
        std_within_hd_64 = np.nanstd(selected_hd_64)
        std_within_hd_48 = np.nanstd(selected_hd_48)
        std_within_hd_32 = np.nanstd(selected_hd_32)

        print("HD 96 within:", mean_within_hd_96, " sd: (", std_within_hd_96, ")")
        print("HD 80 within:", mean_within_hd_80, " sd: (", std_within_hd_80, ")")
        print("HD 64 within:", mean_within_hd_64, " sd: (", std_within_hd_64, ")")
        print("HD 48 within:", mean_within_hd_48, " sd: (", std_within_hd_48, ")")
        print("HD 32 within:", mean_within_hd_32, " sd: (", std_within_hd_32, ")")

        selected_hd_96 = []
        selected_hd_80 = []
        selected_hd_64 = []
        selected_hd_48 = []
        selected_hd_32 = []

        for index in range(len(combined_df)):
            row = combined_df.iloc[index]
            if row["PatchSize"] == 96:
                if (row["fg_ratio"] <= fg_ratio_96.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_96.max()[0]
                ):
                    selected_hd_96.append(row["hausdorff"])
            elif row["PatchSize"] == 80:
                if (row["fg_ratio"] <= fg_ratio_80.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_80.max()[0]
                ):
                    selected_hd_80.append(row["hausdorff"])
            elif row["PatchSize"] == 64:
                if (row["fg_ratio"] <= fg_ratio_64.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_64.max()[0]
                ):
                    selected_hd_64.append(row["hausdorff"])
            elif row["PatchSize"] == 48:
                if (row["fg_ratio"] <= fg_ratio_48.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_48.max()[0]
                ):
                    selected_hd_48.append(row["hausdorff"])
            elif row["PatchSize"] == 32:
                if (row["fg_ratio"] <= fg_ratio_32.min()[0]) or (
                    row["fg_ratio"] >= fg_ratio_32.max()[0]
                ):
                    selected_hd_32.append(row["hausdorff"])

        mean_outside_hd_96 = np.nanmean(selected_hd_96)
        mean_outside_hd_80 = np.nanmean(selected_hd_80)
        mean_outside_hd_64 = np.nanmean(selected_hd_64)
        mean_outside_hd_48 = np.nanmean(selected_hd_48)
        mean_outside_hd_32 = np.nanmean(selected_hd_32)

        std_outside_hd_96 = np.nanstd(selected_hd_96)
        std_outside_hd_80 = np.nanstd(selected_hd_80)
        std_outside_hd_64 = np.nanstd(selected_hd_64)
        std_outside_hd_48 = np.nanstd(selected_hd_48)
        std_outside_hd_32 = np.nanstd(selected_hd_32)

        print("HD 96 outside:", mean_outside_hd_96, " sd: (", std_outside_hd_96, ")")
        print("HD 80 outside:", mean_outside_hd_80, " sd: (", std_outside_hd_80, ")")
        print("HD 64 outside:", mean_outside_hd_64, " sd: (", std_outside_hd_64, ")")
        print("HD 48 outside:", mean_outside_hd_48, " sd: (", std_outside_hd_48, ")")
        print("HD 32 outside:", mean_outside_hd_32, " sd: (", std_outside_hd_32, ")")

        fig, ax = plt.subplots()
        sns.violinplot(
            ax=ax,
            cut=0,
            data=fg_ratio_32,
            palette=["green"],
            inner="points",
            orient="h",
        )
        sns.violinplot(
            ax=ax,
            cut=0,
            data=fg_ratio_64,
            palette=["dodgerblue"],
            inner="points",
            orient="h",
        )
        sns.violinplot(
            ax=ax, cut=0, data=fg_ratio_96, palette=["red"], inner="points", orient="h"
        )
        plt.setp(ax.collections, alpha=0.5)

        # sns.boxplot(ax=ax, data=visualize_df, x="fg_group", y="dsc", hue="PatchSize",
        #                palette=['green','dodgerblue','red'])
        # sns.scatterplot(ax=ax, data=visualize_df, x="fg_group", y="dsc", hue="PatchSize", alpha=0.3,
        #                palette=['green','dodgerblue','red'], size="PatchSize", sizes=[30, 50, 70])

        sns.scatterplot(
            ax=ax,
            data=visualize_df,
            x="fg_ratio",
            y="dsc",
            hue="PatchSize",
            alpha=0.3,
            palette=["green", "dodgerblue", "red"],
            size="PatchSize",
            sizes=[20, 40, 60],
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        plt.grid()
        plt.ylim([0.0, 1.1])
        plt.xlim([0.0, 0.5])
        locs, labels = plt.yticks()  # Get the current locations and labels.
        plt.yticks(
            np.arange(0, 1.2, step=0.2),
            labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        )  # Set label locations.
        plt.xlabel("Foreground voxel proportion")
        plt.ylabel("Dice Similarity Coefficient of test samples")
        plt.title(data_to_view + " (Synthetic task)")
        plt.savefig(
            os.path.join(results_dir, data_to_view + "_synthetic_dice.png"),
            bbox_inches="tight",
        )

        fig, ax = plt.subplots()
        sns.violinplot(
            ax=ax,
            cut=0,
            data=fg_ratio_32,
            palette=["green"],
            inner="points",
            orient="h",
        )
        sns.violinplot(
            ax=ax,
            cut=0,
            data=fg_ratio_64,
            palette=["dodgerblue"],
            inner="points",
            orient="h",
        )
        sns.violinplot(
            ax=ax, cut=0, data=fg_ratio_96, palette=["red"], inner="points", orient="h"
        )
        plt.setp(ax.collections, alpha=0.5)

        # sns.boxplot(ax=ax, data=visualize_df, x="fg_group", y="dsc", hue="PatchSize",
        #                palette=['green','dodgerblue','red'])
        # sns.scatterplot(ax=ax, data=visualize_df, x="fg_group", y="dsc", hue="PatchSize", alpha=0.3,
        #                palette=['green','dodgerblue','red'], size="PatchSize", sizes=[30, 50, 70])
        visualize_df["hausdorff"] = visualize_df["hausdorff"].div(160.0)
        sns.scatterplot(
            ax=ax,
            data=visualize_df,
            x="fg_ratio",
            y="hausdorff",
            hue="PatchSize",
            alpha=0.5,
            palette=["green", "dodgerblue", "red"],
            size="PatchSize",
            sizes=[20, 40, 60],
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        plt.grid()
        plt.ylim([0.0, 1.1])
        plt.xlim([0.0, 0.5])
        locs, labels = plt.yticks()  # Get the current locations and labels.
        plt.yticks(
            np.arange(0, 1.2, step=0.2), labels=["0", "32", "64", "96", "128", "160"]
        )  # Set label locations.
        plt.xlabel("Foreground voxel proportion")
        plt.ylabel("Hausdorff distance of test samples")
        plt.title(data_to_view + " (Synthetic task)")
        plt.savefig(
            os.path.join(results_dir, data_to_view + "_synthetic_hausdorff.png"),
            bbox_inches="tight",
        )

        fig, ax = plt.subplots()
        sns.scatterplot(
            ax=ax,
            data=combined_df,
            x="fg_ratio",
            y="hausdorff",
            hue="PatchSize",
            alpha=0.7,
            palette=["green", "dodgerblue", "orange", "red", "brown"],
            size="PatchSize",
            sizes=[10, 30, 50, 70, 90],
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        plt.axvline(
            x=np.max([fg_ratio_96.min()[0], 0.0]), ymin=0.0, ymax=1.0, color="brown"
        )
        plt.axvline(
            x=np.min([fg_ratio_96.max()[0], 0.5]),
            ymin=0.0,
            ymax=1.0,
            color="brown",
            linestyle="dashed",
        )

        plt.axvline(
            x=np.max([fg_ratio_80.min()[0], 0.0]), ymin=0.0, ymax=1.0, color="red"
        )
        plt.axvline(
            x=np.min([fg_ratio_80.max()[0], 0.5]),
            ymin=0.0,
            ymax=1.0,
            color="red",
            linestyle="dashed",
        )

        plt.axvline(
            x=np.max([fg_ratio_64.min()[0], 0.0]), ymin=0.0, ymax=1.0, color="orange"
        )
        plt.axvline(
            x=np.min([fg_ratio_64.max()[0], 0.5]),
            ymin=0.0,
            ymax=1.0,
            color="orange",
            linestyle="dashed",
        )

        plt.axvline(
            x=np.max([fg_ratio_48.min()[0], 0.0]),
            ymin=0.0,
            ymax=1.0,
            color="dodgerblue",
        )
        plt.axvline(
            x=np.min([fg_ratio_48.max()[0], 0.5]),
            ymin=0.0,
            ymax=1.0,
            color="dodgerblue",
            linestyle="dashed",
        )

        plt.axvline(
            x=np.max([fg_ratio_32.min()[0], 0.0]), ymin=0.0, ymax=1.0, color="green"
        )
        plt.axvline(
            x=np.min([fg_ratio_32.max()[0], 0.5]),
            ymin=0.0,
            ymax=1.0,
            color="green",
            linestyle="dashed",
        )

        plt.grid()
        plt.ylim([0.0, 200.0])
        plt.xlim([-0.02, 0.51])
        plt.xlabel("Foreground voxel proportion")
        plt.ylabel("Hausdorff distance of test samples")
        plt.title(data_to_view)
        plt.savefig(
            os.path.join(
                results_dir, data_to_view + "_foreground_voxel_proportion.png"
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    analyze_synthetic_results()
