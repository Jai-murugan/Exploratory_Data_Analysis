pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Merging and Understanding Data
adult = pd.concat([X,y], axis = 1)

adult.info()
adult.head()
adult.describe()

# Removing useless columns

adult = adult.drop(["fnlwgt", "capital-gain", "capital-loss"], axis = 1)
adult.isnull().sum()
adult[["occupation","native-country"]] = adult[["occupation","native-country"]].replace("?", np.nan)
adult = adult.dropna(subset=['occupation',"native-country"])
adult = adult.reset_index(drop = True)


###########################
# Analyzing by age factor
###########################
age_groups = ["{0} - {1}".format(i, i + 9) for i in range(18, 98, 10)]
age_data = pd.Categorical(age_groups)

adult["income"] = adult["income"].str.replace('.', '', regex=False)
adult["income"].unique()
adult["Age group"] = pd.cut(adult.age, range(18, 101, 10), right=False, labels=age_data)

age_income = adult.groupby(["Age group","income"]).size().unstack(fill_value=0)

age_income['total'] = age_income['<=50K'] + age_income['>50K']
age_income['<=50K_rate'] = (age_income['<=50K'] / age_income['total']).round(2)
age_income['>50K_rate'] = (age_income['>50K'] / age_income['total']).round(2)

#############################
# Visualization
##############################
fig, ax = plt.subplots(figsize=(10, 6))

age_income[['>50K_rate', '<=50K_rate']].plot(kind='bar', stacked=True, ax=ax, color=['lightcoral', 'lightgreen'])

ax.set_xlabel('Age Group')
ax.set_ylabel('Income Rate')
ax.set_title('Income Rate by Age Group')
ax.set_xticklabels(age_income.index, rotation=45)
plt.tight_layout()
plt.show()

###########################
# Analyzing by occupation
###########################
occupation_analysis = adult.groupby(["occupation", "income"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))
occupation_analysis[['>50K', '<=50K']].plot(kind='bar', stacked=True, ax=ax, color=['blue', 'purple'])

ax.set_xlabel('Occupations')
ax.set_ylabel('Income Count')
ax.set_title('Income Distribution by Occupation')
ax.set_xticklabels(occupation_analysis.index, rotation=45, ha="right")

plt.tight_layout()
plt.show()

#######################
# Count of Martial Status
#######################
adult["marital-status"].value_counts()

######################
# Average work hours by country
######################

average_work_hours = []

for (country, income), sub_df in adult.groupby(["native-country", "income"]):
    average_work_hours.append([country, income, round(sub_df["hours-per-week"].mean(), 2)])
average_work_hours = pd.DataFrame(average_work_hours, columns=["Country", "Income", "Mean Hours per Week"])


# The dataset is checked with null values
# The Barplot represents the Income Rate by Age Group
# The second plot represent the Income by type of occupation
# Marital status and Averafe work hours by countries are found





























