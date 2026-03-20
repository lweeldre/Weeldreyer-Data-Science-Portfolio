# Tidy Data Project - Olympics 2008 🏅

- This project demonstrates how to apply tidy data principles using Python. The dataset contains medal counts from the 2008 Olympics, and it is cleaned and transformed into a tidy format for analysis and visualization.

## 📊 Tidy Data Principles:
1) Each variable has its own column
2) Each observation has its own row
3) Each type of observational unit forms a table


- The original dataset was not in a tidy format because multiple variables were stored within the column names instead of being separated into individual columns. To address this issue, the dataset was reshaped using the [melt()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html) function in pandas.
- I used [str.split()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.split.html) to split one column into two new columns.
- I also used [str.replace()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html) to clean up strings with unwanted information.

## 📈 Steps to download the data used in this project:
1) Download the data from the provided [link](https://edjnet.github.io/OlympicsGoNUTS/2008/)
2) Save the file as: olympics_08_medalists.csv
3) Place the file in the same folder as the Jupyter Notebook.
4) Load the dataset in Python using pandas:

-> import pandas as pd

-> import matplotlib.pyplot as plt

-> df_olympics = pd.read_csv("olympics_08_medalists.csv")

  
