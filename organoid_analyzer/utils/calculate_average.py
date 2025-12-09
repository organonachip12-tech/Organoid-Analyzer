import pandas as pd
import numpy as np
from scipy.stats import linregress

def swap_columns(df, col1, col2):
    cols = list(df.columns)
    i, j = cols.index(col1), cols.index(col2)
    cols[i], cols[j] = cols[j], cols[i]
    df = df[cols]
    return df

def calculate_alpha(df):
    alpha_values = []
    for (prefix, track_id), group in df.groupby(["PREFIX", "TRACK_ID"]):

        is_valid = (group["FRAME"] > 0) & (group["MEAN_SQUARE_DISPLACEMENT"] > 0)

        valid_groups = group.loc[is_valid].copy()   # copy avoids the warning
        valid_groups["LAG"] = valid_groups["FRAME"] - valid_groups["FRAME"].iloc[0] + 1

        if len(valid_groups) < 2:  # need at least 2 points to fit
            alpha = np.nan
        else:
            slope, _, _, _, _ = linregress(
                np.log(valid_groups["LAG"]),
                np.log(valid_groups["MEAN_SQUARE_DISPLACEMENT"])
            )
            alpha = slope

        alpha_values.append({
            "PREFIX": prefix,
            "TRACK_ID": track_id,
            "ALPHA": alpha,
            "LABEL": group["LABEL"].iloc[0]
        })

    return pd.DataFrame(alpha_values)

if __name__ == "__main__":
    spots_df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_spot_features.csv"
    track_df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_track_features.csv"
    output_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\averaged_features_by_track_msd.csv"

    merge_track = True
    add_empty_rows_between_prefix = False
    

    columns_to_average = ["AREA", "PERIMETER", "CIRCULARITY", 
                    "ELLIPSE_ASPECTRATIO", 
                    "SOLIDITY", "SPEED", "MEAN_SQUARE_DISPLACEMENT"]
    
    prefixes_to_keep = ["2ND_NYU360", "PDO_Device8", "PDO_Device7", "PDO_Device5", "PDO_Device4", "PDO_Device3",
                        "2ND_NCI8", "CART_NCI2", "CART_NCI6", "CART_NCI8", "CART_NYU360", "PDO_Device6"]


    orig_spots_df = pd.read_csv(spots_df_path) 

    grouped = orig_spots_df.groupby(["PREFIX", "TRACK_ID"])

    averaged = grouped[columns_to_average].mean().reset_index()

    averaged["FRAME_COUNT"] = grouped.size().values

    alpha_df = calculate_alpha(orig_spots_df)

    final_df = pd.merge(averaged, alpha_df, on=["PREFIX", "TRACK_ID"])

    final_df = swap_columns(final_df, "ALPHA", "FRAME_COUNT" )


    if merge_track:
        track_df = pd.read_csv(track_df_path)
        #track_df = track_df.drop(columns=["PREFIX", "TRACK_ID", "LABEL"])

        final_df = pd.merge(final_df, track_df, on=["PREFIX", "TRACK_ID", "LABEL"])


        print(final_df["PREFIX"].str.startswith(tuple(prefixes_to_keep), na=False))

        final_df = final_df[final_df["PREFIX"].str.startswith(tuple(prefixes_to_keep), na=False)]

        if add_empty_rows_between_prefix:
            groups = []
            for prefix, g in final_df.groupby("PREFIX"):
                prefix_split = prefix.split("_")
                
                if prefix_split[2][-1] == "1":
                    groups.append(pd.DataFrame([[""] + [prefix_split[1][:-1] + "_" + prefix_split[1][-1]] + [""] * (len(final_df.columns)-2)], columns=final_df.columns))
                groups.append(pd.DataFrame([[""] + [prefix_split[2]] + [""] * (len(final_df.columns)-2)], columns=final_df.columns))

                columns = ["" if col == "TRACK_ID" else col for col in g.columns]

                groups.append(pd.DataFrame([columns], columns=final_df.columns))
                g.drop(columns="PREFIX", inplace=True, errors='ignore')
                groups.append(g)
                groups.append(pd.DataFrame([[""] * (len(final_df.columns))], columns=final_df.columns))
            
            
            final_df = pd.concat(groups, ignore_index=True)
            final_df = final_df.set_index(final_df.columns[0])

        column_order = [
            "PREFIX",
            "TRACK_ID",
            "AREA",
            "PERIMETER",
            "CIRCULARITY",
            "ELLIPSE_ASPECTRATIO",
            "SOLIDITY",
            "SPEED",
            "ALPHA",
            "TRACK_DISPLACEMENT",
            "TRACK_STD_SPEED",
            "MEAN_DIRECTIONAL_CHANGE_RATE",
            "MEAN_SQUARE_DISPLACEMENT",
            "LABEL",
            "FRAME_COUNT",
            
        ]

        final_df = final_df[column_order]
        print(final_df)
    
    

    print(f"Calulated averages succesfully to {output_path}")
    final_df.to_csv(output_path, index=False)