import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from dabl import clean, plot, detect_types
from dabl.plot import *
import sweetviz as sv


if __name__ == "__main__":

    df_train = pd.read_csv("train.csv", index_col=0)
    
    report_full = sv.analyze(df_train, target_feat='demand')
    report_full.show_html(layout='vertical', filepath='report_full.html')
    
    """Observations from full report:
    1. tourist_attraction and secondary_connectivity are redundant
    2. There are over whelmingly high number of zeros in demand
    3. We need to look at the data in a comparative sense where only positive demand data is assessed against full data
    
    """
    
    comparative_report = sv.compare([df_train, 'full'], [df_train.query("demand>0"), 'positive_demand'], target_feat='demand')
    
    comparative_report.show_html(layout='vertical', filepath='report_comparative.html')
    
    
    