#!/usr/bin/env python3 

# Importing the necessary packages 
import pandas as pd 

# 
class Filtered_dataset():
    def __init__(self, dataset):
        self.df = dataset 

    # 
    def df_filter(self):
        df = self.df 
        # getting the answers for the insured individuals
        is_insured = df[df['Is Insured'] == 1]

        # NUmber of individuals that do not have insurance 
        not_insured = df[df['Is Insured'] == 0]

        # Number of individuals that has Valid MOT 
        has_valid_mot = df[df['Has Valid MOT'] >= 1]

        # Number of individuals that do not have valid mot 
        no_valid_mot = df[df['Has Valid MOT'] == 0]

        # Number of individuals that has valid license 
        valid_license = df[df['Has Valid License'] == 1]

        # Number of individuals that do not have valid license 
        no_valid_license = df[df['Has Valid License'] == 0]

        # Returning the filtered tags 
        return is_insured, not_insured, has_valid_mot, no_valid_mot, valid_license, no_valid_license
