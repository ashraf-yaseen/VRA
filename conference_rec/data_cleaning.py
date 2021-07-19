#%%
import pandas as pd

import glob
#%%
def read_folder(path: str) -> pd.DataFrame:
    """
    Reads all given csv files in a directory and outputs a dataframe of all csv files concatenated together

    Input Args: Path to directory where csv files are stored
    Output: Dataframe containing all data from all input csv files
    """
    files = glob.glob(f"{path}/*.csv") 
    
    dfs = []
    for a_file in files:
        dfs.append(pd.read_csv(a_file, index_col = 0, header = 0))
    
    collective_df = pd.concat(dfs, ignore_index = True)

    return collective_df

def better_dates(df: pd.DataFrame, date_column: str = "Conference Date") -> pd.DataFrame:
    """
    Takes date column and creates two new columns: start date and end date in datetime format for easy usage

    Input Args: Dataframe, Date column of dataframe
    Output: Start Date and End Date datetime columns in a dataframe
    """
    df[["startDate", "endDate"]] = df[date_column].str.split("-", expand = True)
    
    df["endDate"] = pd.to_datetime(df["endDate"])
    df["startDate"] = pd.to_datetime(df["startDate"])

    return df

def unique_confs_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """

    Takes input dataframe and  drops all non-unique entries per each year

    Input: Dataframe
    Output: Dataframe
    """
    NU_frame1 = df.copy()
    NU_frame1["Year"] = NU_frame1["Conference Title"].str.extract(r"(\d{4}) :")
    NU_frame1["Year"] = NU_frame1["Year"].apply(int)
    U_frame = NU_frame1.drop_duplicates(subset = "Conference Title")

    return U_frame

# %%
if __name__ == "__main__":
    wikicfp = read_folder("/workspaces/VRA/conference_rec/wikicfp_csv")
    wikicfp = unique_confs_per_year(wikicfp)
    wikicfp = better_dates(wikicfp)
# %%
