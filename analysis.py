import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the student performance dataset
df = pd.read_csv('StudentsPerformance.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert score columns to numeric
df['math score'] = pd.to_numeric(df['math score'])
df['reading score'] = pd.to_numeric(df['reading score'])
df['writing score'] = pd.to_numeric(df['writing score'])

# Create a total score column
df['total score'] = df['math score'] + df['reading score'] + df['writing score']
df['average score'] = df['total score'] / 3

print("\nUpdated dataset with total and average scores:")
print(df[['math score', 'reading score', 'writing score', 'total score', 'average score']].head())

# =============================================================================
# EXPLORATORY DATA ANALYSIS AND VISUALIZATIONS
# =============================================================================

# 1. Distribution of Scores
plt.figure(figsize=(15, 10))

# Math Score Distribution
plt.subplot(2, 3, 1)
plt.hist(df['math score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Math Scores')
plt.xlabel('Math Score')
plt.ylabel('Frequency')

# Reading Score Distribution
plt.subplot(2, 3, 2)
plt.hist(df['reading score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribution of Reading Scores')
plt.xlabel('Reading Score')
plt.ylabel('Frequency')

# Writing Score Distribution
plt.subplot(2, 3, 3)
plt.hist(df['writing score'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Distribution of Writing Scores')
plt.xlabel('Writing Score')
plt.ylabel('Frequency')

# Average Score Distribution
plt.subplot(2, 3, 4)
plt.hist(df['average score'], bins=20, alpha=0.7, color='gold', edgecolor='black')
plt.title('Distribution of Average Scores')
plt.xlabel('Average Score')
plt.ylabel('Frequency')

# Box plot of all scores
plt.subplot(2, 3, 5)
score_columns = ['math score', 'reading score', 'writing score']
df[score_columns].boxplot(ax=plt.gca())
plt.title('Box Plot of All Scores')
plt.ylabel('Score')
plt.xticks(rotation=45)

# Correlation heatmap
plt.subplot(2, 3, 6)
correlation_matrix = df[['math score', 'reading score', 'writing score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Score Correlations')

plt.tight_layout()
plt.savefig('score_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Gender Analysis
plt.figure(figsize=(15, 8))

# Gender distribution
plt.subplot(2, 3, 1)
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')

# Average scores by gender
plt.subplot(2, 3, 2)
gender_scores = df.groupby('gender')[['math score', 'reading score', 'writing score']].mean()
gender_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Gender')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.legend()

# Math scores by gender
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='gender', y='math score')
plt.title('Math Scores by Gender')

# Reading scores by gender
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='gender', y='reading score')
plt.title('Reading Scores by Gender')

# Writing scores by gender
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='gender', y='writing score')
plt.title('Writing Scores by Gender')

# Total scores by gender
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='gender', y='total score')
plt.title('Total Scores by Gender')

plt.tight_layout()
plt.savefig('gender_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Race/Ethnicity Analysis
plt.figure(figsize=(15, 8))

# Race/ethnicity distribution
plt.subplot(2, 3, 1)
race_counts = df['race/ethnicity'].value_counts()
plt.pie(race_counts.values, labels=race_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Race/Ethnicity Distribution')

# Average scores by race/ethnicity
plt.subplot(2, 3, 2)
race_scores = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean()
race_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Race/Ethnicity')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.legend()

# Math scores by race/ethnicity
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='race/ethnicity', y='math score')
plt.title('Math Scores by Race/Ethnicity')
plt.xticks(rotation=45)

# Reading scores by race/ethnicity
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='race/ethnicity', y='reading score')
plt.title('Reading Scores by Race/Ethnicity')
plt.xticks(rotation=45)

# Writing scores by race/ethnicity
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='race/ethnicity', y='writing score')
plt.title('Writing Scores by Race/Ethnicity')
plt.xticks(rotation=45)

# Total scores by race/ethnicity
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='race/ethnicity', y='total score')
plt.title('Total Scores by Race/Ethnicity')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('race_ethnicity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Parental Education Analysis
plt.figure(figsize=(15, 8))

# Parental education distribution
plt.subplot(2, 3, 1)
edu_counts = df['parental level of education'].value_counts()
plt.pie(edu_counts.values, labels=edu_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Parental Education Distribution')

# Average scores by parental education
plt.subplot(2, 3, 2)
edu_scores = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean()
edu_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Parental Education')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.legend()

# Math scores by parental education
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='parental level of education', y='math score')
plt.title('Math Scores by Parental Education')
plt.xticks(rotation=45)

# Reading scores by parental education
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='parental level of education', y='reading score')
plt.title('Reading Scores by Parental Education')
plt.xticks(rotation=45)

# Writing scores by parental education
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='parental level of education', y='writing score')
plt.title('Writing Scores by Parental Education')
plt.xticks(rotation=45)

# Total scores by parental education
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='parental level of education', y='total score')
plt.title('Total Scores by Parental Education')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('parental_education_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Lunch and Test Preparation Analysis
plt.figure(figsize=(15, 8))

# Lunch distribution
plt.subplot(2, 3, 1)
lunch_counts = df['lunch'].value_counts()
plt.pie(lunch_counts.values, labels=lunch_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Lunch Type Distribution')

# Test preparation distribution
plt.subplot(2, 3, 2)
test_prep_counts = df['test preparation course'].value_counts()
plt.pie(test_prep_counts.values, labels=test_prep_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Test Preparation Distribution')

# Average scores by lunch type
plt.subplot(2, 3, 3)
lunch_scores = df.groupby('lunch')[['math score', 'reading score', 'writing score']].mean()
lunch_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Lunch Type')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.legend()

# Average scores by test preparation
plt.subplot(2, 3, 4)
test_prep_scores = df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
test_prep_scores.plot(kind='bar', ax=plt.gca())
plt.title('Average Scores by Test Preparation')
plt.ylabel('Average Score')
plt.xticks(rotation=0)
plt.legend()

# Total scores by lunch type
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='lunch', y='total score')
plt.title('Total Scores by Lunch Type')

# Total scores by test preparation
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='test preparation course', y='total score')
plt.title('Total Scores by Test Preparation')

plt.tight_layout()
plt.savefig('lunch_test_prep_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# STATISTICAL ANALYSIS AND INSIGHTS
# =============================================================================

print("\n" + "="*80)
print("STATISTICAL ANALYSIS AND INSIGHTS")
print("="*80)

# 1. Overall Performance Statistics
print("\n1. OVERALL PERFORMANCE STATISTICS:")
print("-" * 40)
print(f"Total number of students: {len(df)}")
print(f"Average Math Score: {df['math score'].mean():.2f}")
print(f"Average Reading Score: {df['reading score'].mean():.2f}")
print(f"Average Writing Score: {df['writing score'].mean():.2f}")
print(f"Average Total Score: {df['total score'].mean():.2f}")
print(f"Average Combined Score: {df['average score'].mean():.2f}")

# 2. Gender Performance Analysis
print("\n2. GENDER PERFORMANCE ANALYSIS:")
print("-" * 40)
gender_stats = df.groupby('gender').agg({
    'math score': ['mean', 'std'],
    'reading score': ['mean', 'std'],
    'writing score': ['mean', 'std'],
    'total score': ['mean', 'std']
}).round(2)
print(gender_stats)

# Statistical test for gender differences
from scipy import stats
male_scores = df[df['gender'] == 'male']['total score']
female_scores = df[df['gender'] == 'female']['total score']
t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
print(f"\nT-test for gender differences in total scores:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# 3. Race/Ethnicity Performance Analysis
print("\n3. RACE/ETHNICITY PERFORMANCE ANALYSIS:")
print("-" * 40)
race_stats = df.groupby('race/ethnicity').agg({
    'math score': 'mean',
    'reading score': 'mean',
    'writing score': 'mean',
    'total score': 'mean'
}).round(2)
print(race_stats)

# 4. Parental Education Impact Analysis
print("\n4. PARENTAL EDUCATION IMPACT ANALYSIS:")
print("-" * 40)
edu_stats = df.groupby('parental level of education').agg({
    'math score': 'mean',
    'reading score': 'mean',
    'writing score': 'mean',
    'total score': 'mean'
}).round(2)
print(edu_stats)

# 5. Lunch and Test Preparation Impact
print("\n5. LUNCH AND TEST PREPARATION IMPACT:")
print("-" * 40)
lunch_stats = df.groupby('lunch')['total score'].agg(['mean', 'std', 'count']).round(2)
print("Lunch Type Impact:")
print(lunch_stats)

test_prep_stats = df.groupby('test preparation course')['total score'].agg(['mean', 'std', 'count']).round(2)
print("\nTest Preparation Impact:")
print(test_prep_stats)

# 6. Correlation Analysis
print("\n6. CORRELATION ANALYSIS:")
print("-" * 40)
correlations = df[['math score', 'reading score', 'writing score']].corr()
print("Correlation Matrix:")
print(correlations.round(3))

# 7. Performance Categories
print("\n7. PERFORMANCE CATEGORIES:")
print("-" * 40)
# Create performance categories
def categorize_performance(score):
    if score >= 90:
        return 'Excellent (90-100)'
    elif score >= 80:
        return 'Good (80-89)'
    elif score >= 70:
        return 'Average (70-79)'
    elif score >= 60:
        return 'Below Average (60-69)'
    else:
        return 'Poor (<60)'

df['math_category'] = df['math score'].apply(categorize_performance)
df['reading_category'] = df['reading score'].apply(categorize_performance)
df['writing_category'] = df['writing score'].apply(categorize_performance)
df['overall_category'] = df['average score'].apply(categorize_performance)

print("Math Score Categories:")
print(df['math_category'].value_counts().sort_index())
print("\nReading Score Categories:")
print(df['reading_category'].value_counts().sort_index())
print("\nWriting Score Categories:")
print(df['writing_category'].value_counts().sort_index())
print("\nOverall Performance Categories:")
print(df['overall_category'].value_counts().sort_index())

# 8. Key Insights
print("\n8. KEY INSIGHTS:")
print("-" * 40)
print("• Gender Analysis:")
print(f"  - Female students perform better in reading ({df[df['gender']=='female']['reading score'].mean():.1f}) and writing ({df[df['gender']=='female']['writing score'].mean():.1f})")
print(f"  - Male students perform better in math ({df[df['gender']=='male']['math score'].mean():.1f})")
print(f"  - Overall, {'female' if df[df['gender']=='female']['total score'].mean() > df[df['gender']=='male']['total score'].mean() else 'male'} students have higher total scores")

print("\n• Socioeconomic Factors:")
print(f"  - Students with standard lunch perform {df[df['lunch']=='standard']['total score'].mean() - df[df['lunch']=='free/reduced']['total score'].mean():.1f} points better on average")
print(f"  - Students who completed test preparation score {df[df['test preparation course']=='completed']['total score'].mean() - df[df['test preparation course']=='none']['total score'].mean():.1f} points higher on average")

print("\n• Parental Education Impact:")
best_edu = df.groupby('parental level of education')['total score'].mean().idxmax()
worst_edu = df.groupby('parental level of education')['total score'].mean().idxmin()
print(f"  - Students with {best_edu} parents perform best")
print(f"  - Students with {worst_edu} parents perform worst")

print("\n• Score Correlations:")
print(f"  - Math and Reading correlation: {df['math score'].corr(df['reading score']):.3f}")
print(f"  - Math and Writing correlation: {df['math score'].corr(df['writing score']):.3f}")
print(f"  - Reading and Writing correlation: {df['reading score'].corr(df['writing score']):.3f}")

# 9. Advanced Visualizations
print("\n9. CREATING ADVANCED VISUALIZATIONS...")

# Scatter plot matrix
plt.figure(figsize=(12, 10))
pd.plotting.scatter_matrix(df[['math score', 'reading score', 'writing score']], 
                          alpha=0.6, figsize=(12, 10), diagonal='hist')
plt.suptitle('Scatter Plot Matrix of Test Scores', fontsize=16)
plt.tight_layout()
plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Heatmap of average scores by all categorical variables
plt.figure(figsize=(12, 8))
pivot_data = df.groupby(['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'])['total score'].mean().unstack(fill_value=0)
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Average Total Scores by All Categorical Variables')
plt.tight_layout()
plt.savefig('comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Performance distribution by multiple factors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gender vs Race/Ethnicity
sns.boxplot(data=df, x='race/ethnicity', y='total score', hue='gender', ax=axes[0,0])
axes[0,0].set_title('Total Scores by Race/Ethnicity and Gender')
axes[0,0].tick_params(axis='x', rotation=45)

# Parental Education vs Test Preparation
sns.boxplot(data=df, x='parental level of education', y='total score', hue='test preparation course', ax=axes[0,1])
axes[0,1].set_title('Total Scores by Parental Education and Test Preparation')
axes[0,1].tick_params(axis='x', rotation=45)

# Lunch vs Gender
sns.boxplot(data=df, x='lunch', y='total score', hue='gender', ax=axes[1,0])
axes[1,0].set_title('Total Scores by Lunch Type and Gender')

# Race/Ethnicity vs Test Preparation
sns.boxplot(data=df, x='race/ethnicity', y='total score', hue='test preparation course', ax=axes[1,1])
axes[1,1].set_title('Total Scores by Race/Ethnicity and Test Preparation')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('multi_factor_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("Generated visualizations:")
print("• score_distributions.png - Score distributions and correlations")
print("• gender_analysis.png - Gender-based performance analysis")
print("• race_ethnicity_analysis.png - Race/ethnicity performance analysis")
print("• parental_education_analysis.png - Parental education impact")
print("• lunch_test_prep_analysis.png - Lunch and test preparation impact")
print("• scatter_matrix.png - Scatter plot matrix of test scores")
print("• comprehensive_heatmap.png - Comprehensive performance heatmap")
print("• multi_factor_analysis.png - Multi-factor performance analysis")
print("\nAll analysis complete! Check the generated plots and statistical insights above.")

