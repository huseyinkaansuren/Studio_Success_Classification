import pulldata
import numpy as np
import pandas as pd
from sklearn import preprocessing

pd.set_option("display.max_columns", None)


def preprocessing(df):
    dataframe = df.iloc[:,1:6]
    
    #print(dataframe.head())
    studio_genre = dataframe.iloc[:,:2]
    
    #print(studio_genre.head())
    studio_genre = studio_genre.groupby("Studio", as_index = False).agg(", ".join) #Grouping genres

    #print(studio_genre.head())
    studio_score_rank_pop = dataframe[["Studio", "Score", "Ranked", "Popularity"]] 

    studio_score_rank_pop = studio_score_rank_pop.groupby("Studio", as_index = False).agg({"Score":"mean", "Ranked":"mean", "Popularity":"sum"}) #Grouping all other specs

    #print(studio_score_rank_pop.head())
    studio_df = pd.concat([studio_score_rank_pop, studio_genre.iloc[:,1:]],axis =1)
    
    #print(studio_df)
    
    success = []

    for i in range(len(studio_df)):
        studio_specs = studio_df.loc[i]
    
        if studio_specs["Score"] >8.08:
            success.append("Successful")
        elif 7.9<studio_specs["Score"] <=8.08:
            success.append("Normal")
        else:
            success.append("Unsuccessful")

    studio_df["Success"] = success
    return studio_df
