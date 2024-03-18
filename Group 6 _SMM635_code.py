# encoding: utf-8
import textwrap
import warnings
from datetime import datetime
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from stargazer.stargazer import Stargazer
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# region clean data
def date_parser(datetime_str):
    try:
        return datetime.strptime(datetime_str, '%m/%d/%Y')
    except TypeError:
        return datetime_str


# Load data
date_cols = ['DateofTermination', 'LastPerformanceReview_Date', 'DateofHire']

data = pd.read_csv(
    './dataset/HRDataset_v14.csv.xls',
    date_parser=date_parser,
    parse_dates=date_cols
)

# Assume the latest timestamp as the dataset's date
dataset_time = data[date_cols].max().max()

# Use the date of termination for calculating seniority and age if available, else use the dataset time (the employee is still employed)
reference_time = data['DateofTermination'].where(~data['DateofTermination'].isna(), dataset_time)

# Clean DOB
data['DOB'] = pd.to_datetime(data['DOB'], dayfirst=False, yearfirst=False)
data['DOB'] = data['DOB'].where(data['DOB'] < dataset_time, data['DOB'] - pd.Timedelta(days=365.24 * 100 + 1))

# Normalise to Yes/No
data['HispanicLatino'] = data['HispanicLatino'].str.title()

# Age, seniority, time since perf review
data['Age'] = (reference_time - data['DOB']).dt.days / 365
data['Seniority'] = (reference_time - data['DateofHire']).dt.days / 365
data['Years since last perf review'] = (reference_time - data['LastPerformanceReview_Date']).dt.days / 365

# Fill the missing manager ids with 39 - Webster Butler
data['ManagerID'].fillna(39, inplace=True)

# Correct manager ids
data.loc[data['ManagerName'] == 'Brandon R. LeBlanc', 'ManagerID'] = 1
data.loc[data['ManagerName'] == 'Michael Albert', 'ManagerID'] = 22

# Clean up blank spaces in columns
data['Department'] = data['Department'].str.strip()
data['Sex'] = data['Sex'].str.strip()
data['Position'] = data['Position'].str.strip()

# Clean up miscoded performance score id
# Dee,Randy:
# - Currently employed
# - Performance Score is "Fully Meets"
# - Engagement Survey score and Employment Satisfaction score are high
# Days Late in the last 30 days and the number of Absences are low
# --> The correct PerfScoreID should be 3.
data.loc[data['Employee_Name'] == 'Dee, Randy', 'PerfScoreID'] = 3

# Forrest, Alex:
# - Terminated, reason: Fatal Attraction, Status: Terminated for Cause
# - PerformanceScore is PIP
# --> The correct PerfScoreID should be 1.
# Also has date of performance review after the date of termination
# --> Drop
data = data.query('Employee_Name != "Forrest, Alex"')

data['age_group'] = (data['Age'] - (data['Age'] % 10)).astype(int).astype(str) + 's'
data['Leadership'] = data['Position'].str.lower().str.contains('|'.join(['manager', 'director', 'cio', 'president', 'principal'])).astype(int)

data = data.convert_dtypes()

# Export
data.to_csv('./dataset/cleaned.csv', index=False)
# endregion

# region Q1
df = pd.read_csv('dataset/cleaned.csv')

# Drop Executive Office as there's only 1 person
df = df.query('Department != "Executive Office"')

# 1st plot


def plot_perf_id_by_manager(df, ax):
    PSID_byManager = df.groupby('ManagerName').agg({'PerfScoreID': lambda x: list(x)}).sort_index(ascending=False)['PerfScoreID']
    ax.boxplot(
        PSID_byManager,
        showmeans=True,
        vert=False,
        boxprops={"facecolor": (.4, .6, .8, .5), 'zorder': 10, 'alpha': 1},
        meanprops={'markerfacecolor': '#800000', 'markeredgecolor': '#800000', 'marker': 'o'},
        flierprops={'marker': 'o', 'markerfacecolor': '#486090', 'markeredgecolor': '#486090'},
        zorder=10,
        patch_artist=True,
    )
    ax.set_xlabel('Employee performance', fontweight='bold', fontname="Sans Serif", color="#4d4d4d", size=12)
    ax.set_ylabel('Manager', fontweight='bold', fontname="Sans Serif", color="#4d4d4d", size=12)
    ax.set_title('Employee performance by manager', fontweight='bold', fontname="Sans Serif", color="#4d4d4d", size=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add major gridlines in the y-axis
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='grey', zorder=0)

    # Add xticks
    ax.set_xticks([1, 2, 3, 4], ['1\nPIP', '2\nNeeds Improvement', '3\nFully Meets', '4\nExceeds'])
    ax.tick_params(left=False, bottom=False)

# Modelling


def remove_correlated_variables(df_dummies, threshold=0.8):
    # Absolute value correlation matrix
    corr_matrix = df_dummies.corr().abs()

    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f'There are {len(to_drop)} columns to remove: {to_drop}')

    return df_dummies.drop(to_drop, axis=1)


def check_vif(X):
    vif = pd.Series(
        [
            variance_inflation_factor(X, i)
            for i in range(X.shape[1])
        ],
        index=X.columns,
    )

    print('All VIF < 5: ', all(vif < 5))

    return vif


label_cols = ['PerfScoreID']

numerical_cols = [
    'Salary',
    'EngagementSurvey',
    'EmpSatisfaction',
    'DaysLateLast30',
    'Absences',
    'Seniority',
    'Years since last perf review',
]

cat_cols = [
    'FromDiversityJobFairID',
    'Sex',
    'MaritalDesc',
    'CitizenDesc',
    'HispanicLatino',
    'ManagerName',
    'RecruitmentSource',
    'age_group'
]

data = df[numerical_cols + cat_cols + label_cols]
df_dummies = pd.get_dummies(data, columns=cat_cols, drop_first=True)
df_dummies = remove_correlated_variables(df_dummies)

X = df_dummies.drop(label_cols, axis=1)
Y = df_dummies[label_cols]

X = pd.DataFrame(
    StandardScaler().fit_transform(X),
    columns=X.columns,
    index=X.index,
)

check_vif(X)

model = OrderedModel(Y, X).fit(method='lbfgs', full_output=False)

model_save_dir = Path('./models/Q1')
if not model_save_dir.exists():
    model_save_dir.mkdir(parents=True)

with open(model_save_dir / 'full.txt', 'w') as f:
    f.write(str(model.summary()))

model_coefs = model.conf_int()
model_coefs.columns = ['2.5th percentile', '97.5th percentile']
model_coefs['coeff'] = model.params
model_coefs['pvalues'] = model.pvalues
model_coefs = model_coefs.drop(['1/2', '2/3', '3/4'])

# 2nd plot


def plot_coefs(model_coefs, ax):
    vars_to_plot = [x for x in model_coefs.index if 'ManagerName' in x]
    model_coefs_to_plot = model_coefs.loc[vars_to_plot].sort_index(ascending=False)

    plot_idxs = np.arange(len(model_coefs_to_plot)) + 1

    # Plot the manager variables
    ax.hlines(y=plot_idxs, xmin=model_coefs_to_plot['2.5th percentile'], xmax=model_coefs_to_plot['97.5th percentile'], color='#5e5b5b', linewidth=1.5, zorder=-1)
    ax.scatter(model_coefs_to_plot['2.5th percentile'], plot_idxs, color='#5e5b5b', label='2.5th percentile', marker='|', s=70)
    ax.scatter(model_coefs_to_plot['97.5th percentile'], plot_idxs, color='#5e5b5b', label='97.5th percentile', marker='|', s=70)
    ax.scatter(model_coefs_to_plot['coeff'], plot_idxs, color='#800000', label='coefficient', zorder=10)

    # Add title and axis names
    ax.set_title("Managers do not explain employee performance", fontweight='bold', fontname="Sans Serif", color="#4d4d4d", size=15)
    ax.set_xlabel('\nEstimated relationship', fontweight="bold", fontname="Sans Serif", color="#4d4d4d", size=12)
    ax.set_ylabel('Predictors', fontweight="bold", fontname="Sans Serif", color="#4d4d4d", size=12)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.5, zorder=-1)
    ax.axvline(0, color='black', alpha=0.3, linestyle='-', zorder=-2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, bottom=False)


# Final plot
fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
plot_perf_id_by_manager(df.query('ManagerName != "Alex Sweetwater"'), axs[0])
plot_coefs(model_coefs, axs[1])

vars_to_plot = [x for x in model_coefs.index if 'ManagerName' in x]
yticks = [x.split('_')[-1] for x in vars_to_plot]

axs[0].set_yticks(np.arange(len(yticks)) + 1, yticks)
axs[1].set_ylabel('')
plt.show()

# endregion

# region Q2
df = pd.read_csv("./dataset/cleaned.csv")

# get age for all current employees
df2 = df
df2['DateofTermination'] = df2['DateofTermination'].fillna(value='StillEmployee')
df_age_all = df.groupby(['DateofTermination', 'age_group'])['age_group'].count().unstack(fill_value=0).stack().reset_index(name='numbers')
df_age = df_age_all.loc[df_age_all['DateofTermination'] == 'StillEmployee']


# creating dataframes for factors of diversity
df_gender = df.groupby('Sex')['Sex'].count().reset_index(name='numbers')
df_CitizenDesc = df.groupby('CitizenDesc')['CitizenDesc'].count().reset_index(name='numbers')
df_region = df.groupby('HispanicLatino')['HispanicLatino'].count().reset_index(name='numbers')
df_race = df.groupby('RaceDesc')['RaceDesc'].count().reset_index(name='numbers')

# calculate blaus indices for all diveristy factors
blaus = pd.Series([])


def blaus_factor(df_name):
    sum_numerator = 0
    for index, row in df_name.iterrows():
        numerator = (row['numbers']*(row['numbers']-1))
        sum_numerator += numerator
    sum_denominator = (df_name['numbers'].sum())*(df_name['numbers'].sum()-1)
    return 1 - (sum_numerator/sum_denominator)


blaus['gender'] = blaus_factor(df_gender)
blaus['age'] = blaus_factor(df_age)
blaus['Citizen'] = blaus_factor(df_CitizenDesc)
blaus['region'] = blaus_factor(df_region)
blaus['race'] = blaus_factor(df_race)

# Visualize Diversity using blaus index through radar chart
blausvalues = pd.Series.to_list(blaus)
indices = blaus.index
N = len(indices)
fig0 = plt.figure(figsize=(9, 5.5))
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
ax0 = fig0.add_subplot(1, 1, 1, polar=True)
ax0.set_title('Diversity Indices for factors of diversity', pad=20,  fontweight='bold', fontname="Sans Serif", color="#4d4d4d", size=15)
ax0.set_theta_offset(pi / 2)
ax0.set_theta_direction(-1)
ax0.set_xticks(angles[:-1])
ax0.set_xticklabels(indices, fontname="Sans Serif", color="#4d4d4d", size=10)
plt.xticks(angles[:-1], indices, size=15)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax0.set_rlabel_position(0)
blausvalues += blausvalues[:1]
ax0.plot(angles, blausvalues, linewidth=2, linestyle='solid', label=indices, color='#486090')
ax0.fill(angles, blausvalues, color='#3EBCD2', alpha=0.2)
plt.show()

# dataframe grouped by Race and Gender
df_sex_race = df.groupby(['RaceDesc', 'Sex'])['Sex'].count().unstack(fill_value=0).stack().reset_index(name='RS_numbers')
labels = df.RaceDesc.unique()
dfF = df_sex_race.loc[df_sex_race['Sex'] == 'F']
dfM = df_sex_race.loc[df_sex_race['Sex'] == 'M']

# Visualize Racial and Gender diveristy using Bidirectional bar chart
fig1, ax1 = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)

index = dfM.RaceDesc
column0 = dfF['RS_numbers']
column1 = dfM['RS_numbers']

ax1[0].barh(index, column0, align='center', color='#87CEEB', zorder=10)
ax1[0].set_title('Female', fontsize=18, pad=5, fontweight='bold', fontname="Sans Serif", ha="center", color="#4d4d4d")
ax1[1].barh(index, column1, align='center', color='#0073A1', zorder=10)
ax1[1].set_title('Male', fontsize=18, pad=5, fontweight='bold', fontname="Sans Serif", ha="center", color="#4d4d4d")

ax1[0].set(yticks=index, yticklabels=index)
ax1[0].yaxis.tick_left()
ax1[0].tick_params(axis='y', colors='white', labelcolor='k')  # tick col
ax1[1].tick_params(axis='y', colors='white', labelcolor='k')  # tick col

ax1[1].set_xticks([0, 20, 40, 60, 80, 100])
plt.subplots_adjust(wspace=0, hspace=0)
ax1[0].invert_xaxis()
plt.gca().invert_yaxis()
fig1.tight_layout()
plt.show()

# Visualize age diveristy using pie chart
fig2 = plt.figure(figsize=(15, 10))
ax2 = fig2.add_subplot(111)

labels = df_age['age_group']
theme = plt.get_cmap('Blues')
ax2.set_prop_cycle("color", [theme(1. * i / len(df_age.age_group)) for i in range(len(df_age.age_group))])

patches, labels_, percentages = ax2.pie(
    df_age.numbers,
    startangle=90,
    autopct='%.0f%%',
    textprops={'fontsize': 14, 'color': 'black', 'fontweight': 'bold', 'fontname': "Sans Serif"})
ax2.set_title("Age Group Diversity", weight='bold', fontsize=25, position=(0.5, 1), horizontalalignment='center', verticalalignment='center', fontname="Sans Serif", color="#4d4d4d")
plt.legend(labels, bbox_to_anchor=(1, 1), loc="upper right", fontsize=15)

plt.show()
# endregion

# region Q3
# Load the HRDataset_v14.csv
df = pd.read_csv('./dataset/cleaned.csv')

# # Get gender diverse index
df10 = df.groupby(['RecruitmentSource', 'Sex'])['Sex'].count().unstack(fill_value=0).stack().reset_index(name='numbers')

df11 = df10.loc[df10['RecruitmentSource'] == 'CareerBuilder']
df12 = df10.loc[df10['RecruitmentSource'] == 'Diversity Job Fair']
df13 = df10.loc[df10['RecruitmentSource'] == 'Employee Referral']
df14 = df10.loc[df10['RecruitmentSource'] == 'Google Search']
df15 = df10.loc[df10['RecruitmentSource'] == 'Indeed']
df16 = df10.loc[df10['RecruitmentSource'] == 'LinkedIn']
df17 = df10.loc[df10['RecruitmentSource'] == 'On-line Web application']
df18 = df10.loc[df10['RecruitmentSource'] == 'Other']
df19 = df10.loc[df10['RecruitmentSource'] == 'Website']


# define a function to calculate the diversity index
def blaus_factor(df_name):
    sum_numerator = 0
    for _, row in df_name.iterrows():
        numerator = (row['numbers']*(row['numbers']-1))
        sum_numerator += numerator
    denominator = (df_name['numbers'].sum())*(df_name['numbers'].sum()-1)
    return 1-(sum_numerator/denominator)


gender_blaus = pd.Series([])

# use the function get the diversity index of gender
gender_blaus['df_gender1'] = blaus_factor(df11)
gender_blaus['df_gender2'] = blaus_factor(df12)
gender_blaus['df_gender3'] = blaus_factor(df13)
gender_blaus['df_gender4'] = blaus_factor(df14)
gender_blaus['df_gender5'] = blaus_factor(df15)
gender_blaus['df_gender6'] = blaus_factor(df16)
gender_blaus['df_gender7'] = blaus_factor(df17)
gender_blaus['df_gender8'] = blaus_factor(df18)
gender_blaus['df_gender9'] = blaus_factor(df19)

# # Get race diverse index

df20 = df.groupby(['RecruitmentSource', 'RaceDesc'])['RaceDesc'].count().unstack(fill_value=0).stack().reset_index(name='numbers')

df21 = df20.loc[df20['RecruitmentSource'] == 'CareerBuilder']
df22 = df20.loc[df20['RecruitmentSource'] == 'Diversity Job Fair']
df23 = df20.loc[df20['RecruitmentSource'] == 'Employee Referral']
df24 = df20.loc[df20['RecruitmentSource'] == 'Google Search']
df25 = df20.loc[df20['RecruitmentSource'] == 'Indeed']
df26 = df20.loc[df20['RecruitmentSource'] == 'LinkedIn']
df27 = df20.loc[df20['RecruitmentSource'] == 'On-line Web application']
df28 = df20.loc[df20['RecruitmentSource'] == 'Other']
df29 = df20.loc[df20['RecruitmentSource'] == 'Website']

race_blaus = pd.Series([])

# use the function get the diversity index of race
race_blaus['df_race1'] = blaus_factor(df21)
race_blaus['df_race2'] = blaus_factor(df22)
race_blaus['df_race3'] = blaus_factor(df23)
race_blaus['df_race4'] = blaus_factor(df24)
race_blaus['df_race5'] = blaus_factor(df25)
race_blaus['df_race6'] = blaus_factor(df26)
race_blaus['df_race7'] = blaus_factor(df27)
race_blaus['df_race8'] = blaus_factor(df28)
race_blaus['df_race9'] = blaus_factor(df29)

# # Get age diverse index

df31 = df.groupby(['RecruitmentSource', 'age_group'])['age_group'].count().unstack(fill_value=0).stack().reset_index(name='numbers')

df32 = df31.loc[df31['RecruitmentSource'] == 'CareerBuilder']
df33 = df31.loc[df31['RecruitmentSource'] == 'Diversity Job Fair']
df34 = df31.loc[df31['RecruitmentSource'] == 'Employee Referral']
df35 = df31.loc[df31['RecruitmentSource'] == 'Google Search']
df36 = df31.loc[df31['RecruitmentSource'] == 'Indeed']
df37 = df31.loc[df31['RecruitmentSource'] == 'LinkedIn']
df38 = df31.loc[df31['RecruitmentSource'] == 'On-line Web application']
df39 = df31.loc[df31['RecruitmentSource'] == 'Other']
df40 = df31.loc[df31['RecruitmentSource'] == 'Website']

age_blaus = pd.Series([])

# use the function get the diversity index of age
age_blaus['df_age1'] = blaus_factor(df32)
age_blaus['df_age2'] = blaus_factor(df33)
age_blaus['df_age3'] = blaus_factor(df34)
age_blaus['df_age4'] = blaus_factor(df35)
age_blaus['df_age5'] = blaus_factor(df36)
age_blaus['df_age6'] = blaus_factor(df37)
age_blaus['df_age7'] = blaus_factor(df38)
age_blaus['df_age8'] = blaus_factor(df39)
age_blaus['df_age9'] = blaus_factor(df40)

# # Get region diverse index

df41 = df.groupby(['RecruitmentSource', 'HispanicLatino'])['HispanicLatino'].count().unstack(fill_value=0).stack().reset_index(name='numbers')

df42 = df41.loc[df41['RecruitmentSource'] == 'CareerBuilder']
df43 = df41.loc[df41['RecruitmentSource'] == 'Diversity Job Fair']
df44 = df41.loc[df41['RecruitmentSource'] == 'Employee Referral']
df45 = df41.loc[df41['RecruitmentSource'] == 'Google Search']
df46 = df41.loc[df41['RecruitmentSource'] == 'Indeed']
df47 = df41.loc[df41['RecruitmentSource'] == 'LinkedIn']
df48 = df41.loc[df41['RecruitmentSource'] == 'On-line Web application']
df49 = df41.loc[df41['RecruitmentSource'] == 'Other']
df50 = df41.loc[df41['RecruitmentSource'] == 'Website']

region_blaus = pd.Series([])

# use the function get the diversity index of age
region_blaus['df_region1'] = blaus_factor(df42)
region_blaus['df_region2'] = blaus_factor(df43)
region_blaus['df_region3'] = blaus_factor(df44)
region_blaus['df_region4'] = blaus_factor(df45)
region_blaus['df_region5'] = blaus_factor(df46)
region_blaus['df_region6'] = blaus_factor(df47)
region_blaus['df_region7'] = blaus_factor(df48)
region_blaus['df_region8'] = blaus_factor(df49)
region_blaus['df_region9'] = blaus_factor(df50)

# # Get citizen diverse index

df51 = df.groupby(['RecruitmentSource', 'CitizenDesc'])['CitizenDesc'].count().unstack(fill_value=0).stack().reset_index(name='numbers')

df52 = df51.loc[df51['RecruitmentSource'] == 'CareerBuilder']
df53 = df51.loc[df51['RecruitmentSource'] == 'Diversity Job Fair']
df54 = df51.loc[df51['RecruitmentSource'] == 'Employee Referral']
df55 = df51.loc[df51['RecruitmentSource'] == 'Google Search']
df56 = df51.loc[df51['RecruitmentSource'] == 'Indeed']
df57 = df51.loc[df51['RecruitmentSource'] == 'LinkedIn']
df58 = df51.loc[df51['RecruitmentSource'] == 'On-line Web application']
df59 = df51.loc[df51['RecruitmentSource'] == 'Other']
df60 = df51.loc[df51['RecruitmentSource'] == 'Website']

citizen_blaus = pd.Series([])

# use the function get the diversity index of age
citizen_blaus['df_citizen1'] = blaus_factor(df52)
citizen_blaus['df_citizen2'] = blaus_factor(df53)
citizen_blaus['df_citizen3'] = blaus_factor(df54)
citizen_blaus['df_citizen4'] = blaus_factor(df55)
citizen_blaus['df_citizen5'] = blaus_factor(df56)
citizen_blaus['df_citizen6'] = blaus_factor(df57)
citizen_blaus['df_citizen7'] = blaus_factor(df58)
citizen_blaus['df_citizen8'] = blaus_factor(df59)
citizen_blaus['df_citizen9'] = blaus_factor(df60)

print(citizen_blaus)

# # All diverse index

df61 = gender_blaus.to_frame().reset_index()
df62 = race_blaus.to_frame().reset_index()
df63 = age_blaus.to_frame().reset_index()
df64 = region_blaus.to_frame().reset_index()
df65 = citizen_blaus.to_frame().reset_index()

RecruitmentSource = ['CareerBuilder', 'Diversity Job Fair', 'Employee Referral', 'Google Search', 'Indeed', 'LinkedIn', 'On-line Web application', 'Other', 'Website']
d = {
    'RecruitmentSource': RecruitmentSource,
    'gender': df61[0],
    'race': df62[0],
    'age': df63[0],
    'region': df64[0],
    'citizen': df65[0],
}

df_total_diverse_index = pd.DataFrame(data=d)
df_total_diverse_index

df_total_diverse_index = df_total_diverse_index.drop([df_total_diverse_index.index[6], df_total_diverse_index.index[7]]).reset_index()
df_total_diverse_index = df_total_diverse_index.drop('index', axis=1)
df_total_diverse_index = df_total_diverse_index.drop('region', axis=1)
df_total_diverse_index = df_total_diverse_index.drop('citizen', axis=1)
df_total_diverse_index

# # Plot the radar chart

categories = list(df_total_diverse_index)[1:]
categories

N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
angles

recruitment_source = ['CareerBuilder', 'Diversity Job Fair', 'Employee Referral', 'Google Search', 'Indeed', 'LinkedIn', 'Website']
color = ['green', 'red', 'blue', 'purple', 'orange', 'yellow', 'cyan']

FIG = plt.figure(figsize=(22, 13))
ax = FIG.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories, size=25)

for a in range(8):
    for i, o, n in (zip(recruitment_source, color, range(8))):
        if a == n:
            values = df_total_diverse_index.loc[n].drop('RecruitmentSource').values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=0.8, marker='o', color=o, label=i)
            ax.fill(angles, values, color=o, alpha=0.1)
            n = n + 1
        else:
            pass


ax.set_title("Diversity Indices of Recruitment Sources\n", weight='bold',fontsize = 25, position=(0.5, 1),horizontalalignment='center', verticalalignment='center',
fontname="Sans Serif", color="#4d4d4d")

plt.legend(bbox_to_anchor=(1.3, 1), loc="upper right", fontsize=15)
plt.show()
# endregion

# region Q4

time_cols = ['DateofHire', 'LastPerformanceReview_Date', 'DateofTermination']
df = pd.read_csv('dataset/cleaned.csv', parse_dates=time_cols)

# Modelling

# Drop Executive Office as there's only 1 person.
df = df.query('Department != "Executive Office"')

label_cols = ['Termd']

numerical_cols = [
    'EngagementSurvey',
    'Leadership',
    'Salary',
    'EmpSatisfaction',
    'SpecialProjectsCount',
    'DaysLateLast30',
    'Absences',
    'Seniority',
    'Years since last perf review'
]

cat_cols = [
    'FromDiversityJobFairID',
    'Sex',
    'MaritalDesc',
    'CitizenDesc',
    'HispanicLatino',
    'RaceDesc',
    'Department',
    'RecruitmentSource',
    'PerformanceScore',
    'age_group'
]

data = df[numerical_cols + cat_cols + label_cols]
df_dummies = pd.get_dummies(data, columns=cat_cols, drop_first=True)

input_data = remove_correlated_variables(df_dummies)

# defining the dependent and independent variables
Xtrain = input_data.drop('Termd', axis=1)
ytrain = input_data['Termd']

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # building the model and fitting the data
    model = sm.Logit(ytrain, Xtrain).fit(method='newton')

model_save_dir = Path('./models/Q4')
if not model_save_dir.exists():
    model_save_dir.mkdir(parents=True)

with open(model_save_dir / 'full.txt', 'w') as f:
    f.write(str(model.summary()))


# Select the variables with significant p-values at 0.05 level
logit_sum = model.conf_int()
logit_sum['pvalues'] = model.pvalues
logit_sum['coeff'] = model.params
logit_sum.rename(columns={0: '2.5th percentile', 1: '97.5th percentile'}, inplace=True)
significant_var = logit_sum.query('pvalues < 0.05')
significant_var = significant_var.reset_index()

significant_var['factors'] = significant_var['index'].str.split('_').str[0]
significant_var.loc[~significant_var['factors'].isin(['MaritalDesc', 'RecruitmentSource', 'age', 'PerformanceScore']), 'factors'] = 'Others'
significant_var.loc[significant_var['factors'] == 'age', 'factors'] = 'Age Group'
significant_var.loc[significant_var['factors'] == 'PerformanceScore', 'factors'] = 'PerfScore'

significant_var['sub_group'] = significant_var['index'].str.split('_').str[-1]

# bottom plot
# Visualise
factor_counts = significant_var.groupby('factors').size()
factor_counts.name = 'factor_count'
factor_counts = factor_counts[significant_var['factors'].unique()].reset_index()
factor_counts['color'] = ['#486090', '#FF6347', '#808000', '#FFA500', '#9A607F']


def plot_predictors(significant_var, ax):
    # Data
    x = significant_var['sub_group']

    # Labels
    labels = ['Engagement\nSurvey', 'Leadership', 'DaysLate\nLast30', 'Seniority', 'Years since\nlast perf\nreview', 'Married', 'Separated', 'Single',
              'Employee\nReferral', 'Indeed', 'LinkedIn', 'Needs\nImprovement', 'PIP', '30s', '40s', '50s', '60s']

    # The vertical plot is made using the vline function
    ax.vlines(x=x, ymin=significant_var['2.5th percentile'], ymax=significant_var['97.5th percentile'], color='grey', alpha=1, linewidth=2, zorder=-1)
    ax.scatter(x, significant_var['97.5th percentile'], color='grey', s=5, marker='o')
    ax.scatter(x, significant_var['2.5th percentile'], color='grey', s=5, marker='o')
    ax.scatter(x, significant_var['coeff'], color=np.where(significant_var['2.5th percentile'] >= 0, '#6495ED', '#800000'), zorder=10)

    # Add title and axis names
    ax.set_title("Significant predictors for job termination\n", loc='center', fontsize=18, fontweight="bold", fontname="Sans Serif", color="#4d4d4d")
    ax.set_xlabel('Predictors', fontweight="bold", labelpad=40, fontname="Sans Serif", color="#4d4d4d", fontsize=12)
    ax.set_ylabel('Estimated relationship', fontweight="bold", fontname="Sans Serif", color="#4d4d4d", fontsize=12)
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.5, zorder=-2)
    ax.axhline(0, color='red', alpha=0.3, linestyle='-')

    ax.set_xticks(np.arange(17), labels, wrap=True)

    # Remove the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Group factors
    text_start_pos = -0.5
    factor_pos = -48

    for factor, factor_count, color in factor_counts.values:
        ax.text(
            text_start_pos + factor_count / 2, factor_pos, factor.upper(),
            color=color,
            weight="bold",
            ha="center",
            va="bottom",
            fontname="Sans Serif",
            fontsize=10,
        )

        text_start_pos += factor_count

    tick_colors = sum([[color] * factor_count for _, factor_count, color in factor_counts.values], [])

    for ticklabel, tickcolor in zip(ax.get_xticklabels(), tick_colors):
        ticklabel.set_color(tickcolor)


# Reason for termination
termr = df.query('TermReason != "N/A-StillEmployed"')['TermReason'].value_counts().reset_index()
termr['Group'] = termr['index'].where(termr['TermReason'] > 4, 'Others')

group_termr = termr.groupby('Group')['TermReason'].sum().reset_index().sort_values(by='TermReason', ascending=False)

terma = df.query('EmploymentStatus != "Active"')['EmploymentStatus'].value_counts().reset_index()
terma['Status'] = ['Voluntarily', 'For Cause']

# 2nd plot


def plot_reasons(group_termr, terma, ax):
    labels = group_termr['Group']
    sizes = group_termr['TermReason']
    colors = ['#A6ABAD', '#00587A', '#0073A1', '#00A1E0', '#00BCE3', '#87CEEB', '#89BCC4', '#9BD3DD', '#A4E0EB']

    patches, labels_, percentages = ax.pie(
        sizes, colors=colors,
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'color': 'white', 'fontweight': 'bold', 'fontname': "Sans Serif"},
        startangle=90, frame=True,
        autopct="%.2f%%",
        pctdistance=0.85,
    )

    ax.axis('off')

    ax.add_artist(plt.Circle((0, 0), 0.6, color='white', linewidth=0))

    # Adding Title of chart
    ax.set_title('Why Employees left the Company', fontweight='bold', size=15, fontname="Sans Serif", ha="center", color="#4d4d4d")

    ax.legend(labels, loc='upper right', bbox_to_anchor=(1.35, 0.75))


# Attrition rate
def attrition_rate_by(df, freq):
    attrition_df = df.copy()

    # Timestamps for slicing the data to compute attrition rate
    start_time = attrition_df[time_cols].min().min()
    end_time = attrition_df[time_cols].max().max()

    timestamps = [start_time] + pd.date_range(start_time, end_time, freq=freq).tolist() + [end_time]

    # Fill DoT for ease of computing
    attrition_df['DateofTermination'] = attrition_df['DateofTermination'].fillna(pd.to_datetime('2020-01-01'))

    employees_stats = []
    for i in range(len(timestamps) - 1):

        period_start = timestamps[i] + pd.Timedelta(days=1)
        period_end = timestamps[i + 1]

        # Equals to the number of employees hired minus terminated before the period start
        starting_employee_count = (attrition_df['DateofHire'] < period_start).sum() - (attrition_df['DateofTermination'] < period_start).sum()

        employees_hired = np.sum((attrition_df['DateofHire'] >= period_start) & (attrition_df['DateofHire'] <= period_end))
        employees_left = np.sum((attrition_df['DateofTermination'] >= period_start) & (attrition_df['DateofTermination'] <= period_end))

        ending_employee_count = starting_employee_count + employees_hired - employees_left

        # Number of employees left divided by the average number of employees during the period
        attrition_rate = employees_left / ((starting_employee_count + ending_employee_count) / 2) * 100

        employees_stats.append({
            'year': period_end.year,
            'period_start': period_start,
            'period_end': period_end,
            'starting_employee_count': starting_employee_count,
            'ending_employee_count': ending_employee_count,
            'employees_hired': employees_hired,
            'employees_left': employees_left,
            'attrition_rate': attrition_rate,
        })

    employees_stats_df = pd.DataFrame(employees_stats)

    if freq == 'Q':
        employees_stats_df['quarter'] = employees_stats_df['period_end'].dt.quarter

    return employees_stats_df


attrition_rate_by_quarter = attrition_rate_by(df, 'Q')
attrition_rate_by_year = attrition_rate_by(df, 'Y')

# 1st plot


def plot_attrition(attrition_rate_by_year, ax):

    x_data = np.arange(len(attrition_rate_by_year['period_end']))

    ax.set_zorder(10)
    ax.patch.set_visible(False)

    ax.plot(x_data, attrition_rate_by_year['attrition_rate'], zorder=10, c='#800000', label='Attrition Rate', marker='d')
    ax.set_title('Attrition rate by Year', fontweight='bold', size=15, fontname="Sans Serif", ha="center", color="#4d4d4d")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Attrition rate (%)')

    for x, y in zip(x_data, attrition_rate_by_year['attrition_rate']):
        ax.annotate('{:.1f}%'.format(y), (x, y), ha='center', textcoords="offset points", xytext=(0, 10), zorder=10, fontweight='bold')

    axs2 = ax.twinx()

    axs2.set_zorder(1)

    bar_width = 0.25

    axs2.bar(x_data - bar_width / 2, attrition_rate_by_year['employees_hired'], width=bar_width, zorder=1, label='Hired', color='#3EBCD2')
    axs2.bar(x_data + bar_width / 2, attrition_rate_by_year['employees_left'], width=bar_width, zorder=1, label='Left', color='#DB444B')

    axs2.grid(axis='y', color='grey', linestyle='-', linewidth=0.5, zorder=-2)
    axs2.spines['top'].set_visible(False)
    axs2.spines['bottom'].set_visible(False)
    axs2.spines['left'].set_visible(False)

    axs2.set_ylabel('Employee count', rotation=270, labelpad=20)

    plt.legend()

    # Ticks
    ax.set_xticks(x_data, attrition_rate_by_year['year'], rotation=45)


# Final combined plot
fig = plt.figure(figsize=(25, 14))
fig.subplots_adjust(wspace=0, hspace=0.05)

subfigs = fig.subfigures(2, 1)
axs_up = subfigs[0].subplots(1, 2)
plot_attrition(attrition_rate_by_year, axs_up[0])
plot_reasons(group_termr, terma, axs_up[1])

axs_down = subfigs[1].subplots()

plot_predictors(significant_var, axs_down)
plt.show()

# endregion

# region Q5
# load data
DF = pd.read_csv('dataset/cleaned.csv')

# Rename a column
DF.rename(columns={"Years since last perf review": "YearsSinceLastPerfReview"}, inplace=True)

# ### Check Outlier
# Remove the Deparment Executive Office from the analysis as this is the CEO and earns a lot more than the rest
# Remove the executive officer who is the CEO and earns a lot more than the rest
DF.drop(DF[DF['Department'] == "Executive Office"].index, axis=0, inplace=True)
DF.reset_index(inplace=True)

# Dictionary of professions and counts, thresholded >= 10
proff_count = dict([
    (prof, len(list(DF.loc[DF["Position"] == prof, "Position"])))
    for prof
    in np.unique(list(DF["Position"])) if len(list(DF.loc[DF["Position"] == prof, "Position"])) >= 10
])

# New DF where only the above professions are present
# convert the dataframe to use the position as the index
# use the names of the unique professions to get the specific rows - using loc
DF_10 = DF.set_index('Position').loc[list(proff_count.keys())].reset_index(inplace=False)

# ## Exploratory Data Analysis
# Salary vs. Department

# department vs salary table
department_table = pd.pivot_table(data=DF, index="Department", values="Salary", aggfunc=[np.mean, np.median, 'count'])
department_table

salary_by_dept = []
df = DF.copy()
for dept in df["Department"].unique():
    salary_by_dept.append(list(df.loc[df["Department"] == dept, "Salary"]))


df_dept = DF.copy().set_index("Department")

# **Correlation between numerical variables**
# Pick only the numerical columns
data_frame = DF[["Salary", "SpecialProjectsCount", "DaysLateLast30", "Absences",
                 "EmpSatisfaction", "EngagementSurvey", "PerfScoreID",
                 "GenderID", "MarriedID", "Age", "Seniority", "YearsSinceLastPerfReview"]]

# Compute correlation
pear_corr = data_frame.corr(method='pearson')

# ## Regression based analysis
# We now perform regression based analysis to check if there is a lack of equity in pay
# after accounting for controls such as department/job title, seniority, performance
# score and engagement survey score.
# These controls were chosen after checking for correlations in the correlation matrix.

# Perform Regression Analysis
# ----------------------------
# model 1 - Personal characteristics
fml = "np.log(Salary) ~ RaceDesc + Sex + HispanicLatino + CitizenDesc + MaritalDesc + age_group"
model1 = smf.ols(formula=fml, data=DF).fit()
print("==============================================================================")
print("================================  MODEL 1  ===================================")
print("==============================================================================")
print(model1.summary())

# model 2 - employee level individual controls
fml = "np.log(Salary) ~ PerformanceScore + Seniority + age_group + YearsSinceLastPerfReview + EngagementSurvey + EmpSatisfaction + Absences + SpecialProjectsCount + RaceDesc + Sex + HispanicLatino + CitizenDesc + MaritalDesc"
model2 = smf.ols(formula=fml, data=DF).fit()
print("==============================================================================")
print("================================  MODEL 2  ===================================")
print("==============================================================================")
print(model2.summary())

# model 3 - include position control
fml = "np.log(Salary) ~ Position + PerformanceScore + Seniority + age_group + YearsSinceLastPerfReview + EngagementSurvey + EmpSatisfaction + Absences + SpecialProjectsCount + RaceDesc + Sex + HispanicLatino + CitizenDesc + MaritalDesc"
model3 = smf.ols(formula=fml, data=DF).fit()
print("==============================================================================")
print("================================  MODEL 3  ===================================")
print("==============================================================================")
print(model3.summary())

# Model 1: Only personal characteristics of an employee such as Sex, Race,
# Hispanic/Latino ethnicity and US Citizenship status, Marital status and Age group are
# used as independent variables to predict salary.

# Model 2: Individual characteristics like performance score, seniority, employee
# satisfaction, absences, number of special projects undertaken, years since last
# performance review and employee engagement survey score are added as controls.

# Model 3: Job Position is added as an additional control to check for an evidence of
# pay gap after adjusting for all factors that typically influence salary.

# Model 3 explains $ 85\% $ of the variance in the data (adjusted $\text{R}^2$) and has
# a p-value that is significant. We however, do not see any significant linear
# relationship between salary and any personal characteristics of an employee, as the
# p-values associated with these variables are high. Therefore we fail to reject the
# null hypothesis that there is no relationship between salary and personal
# characteristics such as Sex, race, ethnicity, citizenship, age group and marital
# status. The slightly positive (0.3%) relation between salary and absences is seen in
# the above exploratory analysis where people with high salaries had a high number of
# absences.

# These results indicate that there is no evidence of a pay gap with respect to personal
# characteristics like gender, race, ethnicity, citizenship, age and marital status of
# employees.

# Code to summarize the above data in a cleaner format
results = Stargazer([model1, model2, model3])

# Summarize the results in a clean way
results.title('Categorical Explanatory Regression To Check Pay Gaps')
results.custom_columns(['Model 1', 'Model 2', 'Model 3'], [1, 1, 1])
results.show_model_numbers(False)
results.covariate_order([
    'Sex[T.M]',
    'RaceDesc[T.Asian]',
    'RaceDesc[T.Black or African American]',
    'RaceDesc[T.Hispanic]',
    'RaceDesc[T.Two or more races]',
    'RaceDesc[T.White]',
    'HispanicLatino[T.Yes]',
    'CitizenDesc[T.Non-Citizen]',
    'CitizenDesc[T.US Citizen]',
    'MaritalDesc[T.Married]',
    'MaritalDesc[T.Separated]',
    'MaritalDesc[T.Single]',
    'MaritalDesc[T.Widowed]',
    'age_group[T.30s]',
    'age_group[T.40s]',
    'age_group[T.50s]',
    'age_group[T.60s]',
    'EngagementSurvey',
    'SpecialProjectsCount',
    'Absences',
    'Seniority',
    'YearsSinceLastPerfReview'
])
results.rename_covariates({
    'Sex[T.M]': 'Male',
    'perfEval': 'Performance Evaluation',
    'HispanicLatino[T.Yes]': 'Hispanic or Latino',
    'CitizenDesc[T.Non-Citizen]': 'Non-Citizen',
    'CitizenDesc[T.US Citizen]': 'US Citizen',
    'RaceDesc[T.Asian]': 'Race - Asian',
    'RaceDesc[T.Black or African American]': 'Race - Black or African American',
    'RaceDesc[T.Hispanic]': 'Race - Hispanic',
    'RaceDesc[T.Two or more races]': 'Race - Two or more Races',
    'RaceDesc[T.White]': 'Race - White',
    'MaritalDesc[T.Married]': "Married",
    'MaritalDesc[T.Separated]': "Separated",
    'MaritalDesc[T.Single]': "Single",
    'MaritalDesc[T.Widowed]': "Widowed",
    'age_group[T.30s]': "Age Group - 30s",
    'age_group[T.40s]': "Age Group - 40s",
    'age_group[T.50s]': "Age Group - 50s",
    'age_group[T.60s]': "Age Group - 60s",
    "YearsSinceLastPerfReview": "Years since last perf review"
})
results.add_line('Controls:', [' ', ' ', ' '])
results.add_line('Performance Evaluation', ['X', u'\u2713', u'\u2713'])
results.add_line('Engagement Survey Score', ['X', u'\u2713', u'\u2713'])
results.add_line('Seniority', ['X', u'\u2713', u'\u2713'])
results.add_line('Absences', ['X', u'\u2713', u'\u2713'])
results.add_line('SpecialProjectsCount', ['X', u'\u2713', u'\u2713'])
results.add_line('Years since last perf review', ['X', u'\u2713', u'\u2713'])
results.add_line('Job Position', ['X', 'X', u'\u2713'])
results.show_degrees_of_freedom(False)
results

# Anova table
table = sm.stats.anova_lm(model3, typ=2)
print(table)

# **Residual Plot analysis** Plot shows that the model is a good fit as the residuals
# are randomly scattered around zero with a constant variance. The plot of predicted vs
# true salaries also shows that the model has decent prediction accuracy.

# ### Visualization
# #### salary, race and gender
a = np.round(DF_10.groupby(["Position"])["Salary"].aggregate(["mean", "count"]), 2)
a.reset_index(inplace=True)

b = np.round(DF_10.groupby(["Position", "RaceDesc", "Sex"])["Salary"].aggregate(["mean", "count"]), 2)
positions = DF_10["Position"].unique()
races = DF_10["RaceDesc"].unique()


def count_to_size(val):
    max_sz = 20
    min_sz = 3
    return np.sqrt((val - 1) / (80 - 1)) * (max_sz - min_sz) + min_sz


pos_colors = ["#486090", "#9CCCCC", "#D7BFA6", "#7890A8", "#C7B0C1", "#6078A8"]
race_markers = ["o", "o", "o", "o", "o", "o"]
race_colors = ["#800000", "#FF6347", "#6495ED", "#FFA500", "#808000", "#FF1493"]
race_style = ["dotted", "solid", "dashed", "dashdot", ""]

fig, ax = plt.subplots(figsize=(12, 5))

major_offset = 0.0
y_prev = 0
for j, pos in enumerate(positions):
    per_position = b.loc[pos]
    per_position.reset_index(inplace=True)
    per_position.set_index("RaceDesc", inplace=True)

    y_mean = a.loc[a["Position"] == pos, "mean"]

    pos_color = pos_colors[j]
    plt.hlines([y_mean, y_mean], major_offset - 1.4, major_offset + 1.4, color=pos_color)

    minor_offset = -1.0
    for i, race in enumerate(races):
        race_marker = race_markers[i]
        race_color = race_colors[i]

        if race in per_position.index:
            # print(per_position.loc[race, "Sex"].unique())
            sexes = np.unique(list(per_position.loc[race, "Sex"]))
            per_position_per_race = per_position.loc[race]

            if len(sexes) > 1:
                # circle for salary of men
                if "M" in sexes:
                    x = major_offset + minor_offset
                    y_circ = per_position_per_race.loc[per_position_per_race["Sex"] == "M", "mean"]

                    # line from mean line to circle
                    plt.vlines([x, x], y_mean, y_circ, color=race_color)

                    # circle for salary of men
                    sz = count_to_size(per_position_per_race.loc[per_position_per_race["Sex"] == "M", "count"][0])
                    plt.plot(x, y_circ, mfc=race_color, mec=race_color,
                             marker=race_marker, markersize=sz, alpha=1)

                # circle for salary of women
                if "F" in sexes:
                    x = major_offset + minor_offset
                    y_circ = per_position_per_race.loc[per_position_per_race["Sex"] == "F", "mean"]

                    # line from meana line to circle
                    plt.vlines([x, x], y_mean, y_circ, color=race_color)

                    sz = count_to_size(per_position_per_race.loc[per_position_per_race["Sex"] == "F", "count"][0])
                    plt.plot(x, y_circ, mfc="w", mec=race_color,
                             marker=race_marker, markersize=sz, alpha=1)
            else:
                sz = count_to_size(per_position_per_race["count"])

                x = major_offset + minor_offset
                y_circ = per_position_per_race["mean"]

                # line from mean line to circle
                plt.vlines([x, x], y_mean, y_circ, color=race_color)

                if per_position_per_race["Sex"] == "M":
                    plt.plot(x, y_circ, mfc=race_color, mec=race_color,
                             marker=race_marker, markersize=sz, alpha=1)
                else:
                    plt.plot(x, y_circ, mfc="w", mec=race_color, marker=race_marker, markersize=sz, alpha=1)

        minor_offset += 0.4

    # draw vertical line connecting the horizontal lines
    if j > 0:
        plt.vlines([major_offset - 1.4], y_prev, y_mean, color="#4d4d4d", linewidth=0.3)

    # draw heading box
    plt.text(
        major_offset, 115000, textwrap.fill(pos, 14),
        color=pos_color,
        weight="bold",
        ha="center",
        va="bottom",
        fontname="Sans Serif",
        fontsize=10,
        bbox=dict(
            facecolor="none",
            edgecolor=pos_color,
            linewidth=1,
            boxstyle="round",
            pad=0.2
        )
    )

    y_prev = y_mean
    major_offset += 2.8

# Hide border spines
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["bottom"].set_color("none")
ax.spines["left"].set_color("none")

# Remove X - ticks and legends
plt.xticks([], "")

# Custom Legend
circles_x = [-0.75] * 6
circles_y = [106000, 104000, 102000, 100000, 98000, 96000]
for i in range(len(circles_x)):
    plt.plot(circles_x[i], circles_y[i], "o", markersize=5, c=race_colors[i])
    plt.text(circles_x[i] + 0.2, circles_y[i], races[i], fontname="Sans Serif",
             fontsize=8, color="#4d4d4d", va="center")
plt.plot(-0.75, 94000, "o", markersize=5, mfc='#4d4d4d', mec="#4d4d4d")
plt.text(-0.75 + 0.2, 94000, "Male", fontname="Sans Serif", fontsize=8, color="#4d4d4d", va="center")
plt.plot(0.2, 94000, "o", markersize=5, mec='#4d4d4d', mfc="w")
plt.text(0.2 + 0.2, 94000, "Female", fontname="Sans Serif", fontsize=8, color="#4d4d4d", va="center")

# legend for marker size
plt.text(11.2, 57000, "Count of Employees", fontname="Sans Serif", fontsize=8, ha="center")
marker_x = [10.6, 11, 11.4, 11.9]
marker_count = [1, 15, 40, 80]
marker_size = count_to_size(np.array(marker_count))
for i in range(len(marker_x)):
    plt.plot(marker_x[i], 54000, markersize=marker_size[i], marker="o", color="#4d4d4d")
    plt.text(marker_x[i], 49000, str(marker_count[i]), ha="center", fontsize=8, color="#4d4d4d")
ax.add_patch(plt.Rectangle((10.2, 48000), 2, 11000, fill=False))
plt.text(5, 125000, "Pay Equity Analysis - Gender and Race", ha="center", color="#4d4d4d",
         fontsize=12, weight="bold", fontname="Sans Serif")

# Set Y-ticks
_ = plt.yticks([50000, 60000, 70000, 80000, 90000, 100000, 110000], color="#4d4d4d")
# Set Y label
_ = plt.ylabel("Mean Salary", fontname="Sans Serif", fontsize=10, color="#4d4d4d")

plt.show()

# #### salary, ethnicity and gender
a = np.round(DF_10.groupby(["Position"])["Salary"].aggregate(["mean", "count"]), 2)
a.reset_index(inplace=True)

b = np.round(DF_10.groupby(["Position", "HispanicLatino", "Sex"])["Salary"].aggregate(["mean", "count"]), 2)
positions = DF_10["Position"].unique()
eths = DF_10["HispanicLatino"].unique()

pos_colors = ["#486090", "#9CCCCC", "#D7BFA6", "#7890A8", "#C7B0C1", "#6078A8"]
eth_markers = ["o", "o", "o", "o", "o", "o"]
eth_colors = ["#800000", "#FF6347", "#6495ED", "#FFA500", "#808000", "#FF1493"]
eth_style = ["dotted", "solid", "dashed", "dashdot", ""]

fig, ax = plt.subplots(figsize=(12, 5))

major_offset = 0
y_prev = 0
for j, pos in enumerate(positions):
    per_position = b.loc[pos]
    per_position.reset_index(inplace=True)
    per_position.set_index("HispanicLatino", inplace=True)

    y_mean = a.loc[a["Position"] == pos, "mean"]

    pos_color = pos_colors[j]
    plt.hlines([y_mean, y_mean], major_offset - 1.4, major_offset + 1.4, color=pos_color)

    minor_offset = 0.93 - 1.4
    for i, eth in enumerate(eths):
        eth_marker = eth_markers[i]
        eth_color = eth_colors[i]

        if eth in per_position.index:
            sexes = np.unique(list(per_position.loc[eth, "Sex"]))
            per_position_per_eth = per_position.loc[eth]

            if len(sexes) > 1:
                if "M" in sexes:
                    x = major_offset + minor_offset
                    y_circ = per_position_per_eth.loc[per_position_per_eth["Sex"] == "M", "mean"]

                    # line from mean line to circle
                    plt.vlines([x, x], y_mean, y_circ, color=eth_color)

                    # circle for salary of men
                    sz = count_to_size(per_position_per_eth.loc[per_position_per_eth["Sex"] == "M", "count"][0])
                    plt.plot(x, y_circ, mfc=eth_color, mec=eth_color,
                             marker=eth_marker, markersize=sz, alpha=1)

                # circle for salary of women
                if "F" in sexes:
                    x = major_offset + minor_offset
                    y_circ = per_position_per_eth.loc[per_position_per_eth["Sex"] == "F", "mean"]

                    # line from meana line to circle
                    plt.vlines([x, x], y_mean, y_circ, color=eth_color)

                    sz = count_to_size(per_position_per_eth.loc[per_position_per_eth["Sex"] == "F", "count"][0])
                    plt.plot(x, y_circ, mfc="w", mec=eth_color,
                             marker=eth_marker, markersize=sz, alpha=1)
            else:
                sz = count_to_size(per_position_per_eth["count"])

                x = major_offset + minor_offset
                y_circ = per_position_per_eth["mean"]

                # line from mean line to circle
                plt.vlines([x, x], y_mean, y_circ, color=eth_color)

                if per_position_per_eth["Sex"] == "M":
                    plt.plot(x, y_circ, mfc=eth_color, mec=eth_color,
                             marker=eth_marker, markersize=sz, alpha=1)
                else:
                    plt.plot(x, y_circ, mfc="w", mec=eth_color, marker=eth_marker, markersize=sz, alpha=1)

        minor_offset += 0.93

    # draw vertical line connecting the horizontal lines
    if j > 0:
        plt.vlines([major_offset - 1.4], y_prev, y_mean, color="#4d4d4d", linewidth=0.3)

    # draw heading box
    plt.text(
        major_offset, 115000, textwrap.fill(pos, 14),
        color=pos_color,
        weight="bold",
        ha="center",
        va="bottom",
        fontname="Sans Serif",
        fontsize=10,
        bbox=dict(
            facecolor="none",
            edgecolor=pos_color,
            linewidth=1,
            boxstyle="round",
            pad=0.2
        )
    )

    y_prev = y_mean
    major_offset += 2.8

# Hide border spines
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.spines["bottom"].set_color("none")
ax.spines["left"].set_color("none")

# Remove X - ticks and legends
plt.xticks([], "")

# Custom Legend
circles_x = [-0.75] * 2
circles_y = [106000, 104000]
circle_labels = ["Not Hispanic/Latino", "Hispanic/Latino"]
for i in range(len(circles_x)):
    plt.plot(circles_x[i], circles_y[i], "o", markersize=5, c=eth_colors[i])
    plt.text(circles_x[i] + 0.2, circles_y[i], circle_labels[i], fontname="Sans Serif",
             fontsize=8, color="#4d4d4d", va="center")
plt.plot(-0.75, 102000, "o", markersize=5, mfc='#4d4d4d', mec="#4d4d4d")
plt.text(-0.75 + 0.2, 102000, "Male", fontname="Sans Serif", fontsize=8, color="#4d4d4d", va="center")
plt.plot(-0.75, 100000, "o", markersize=5, mec='#4d4d4d', mfc="w")
plt.text(-0.75 + 0.2, 100000, "Female", fontname="Sans Serif", fontsize=8, color="#4d4d4d", va="center")

# legend for marker size
plt.text(11.2, 57000, "Count of Employees", fontname="Sans Serif", fontsize=8, ha="center")
marker_x = [10.6, 11, 11.4, 11.9]
marker_count = [1, 15, 40, 80]
marker_size = count_to_size(np.array(marker_count))
for i in range(len(marker_x)):
    plt.plot(marker_x[i], 54000, markersize=marker_size[i], marker="o", color="#4d4d4d")
    plt.text(marker_x[i], 49000, str(marker_count[i]), ha="center", fontsize=8, color="#4d4d4d")
ax.add_patch(plt.Rectangle((10.2, 48000), 2, 11000, fill=False))
plt.text(5, 125000, "Pay Equity Analysis - Gender and Ethnicity", ha="center", color="#4d4d4d",
         fontsize=12, weight="bold", fontname="Sans Serif")

# Set Y-ticks
_ = plt.yticks([50000, 60000, 70000, 80000, 90000, 100000, 110000], color="#4d4d4d")
# Set Y label
_ = plt.ylabel("Mean Salary", fontname="Sans Serif", fontsize=10, color="#4d4d4d")

plt.show()
# endregion
