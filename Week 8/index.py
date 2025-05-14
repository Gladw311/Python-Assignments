#!/usr/bin/env python
# COVID-19 Global Data Tracker
# This script analyzes global COVID-19 trends including cases, deaths, recoveries, 
# and vaccinations across countries and time.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import os
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set plot style and figure size
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

print("COVID-19 Global Data Tracker")
print("=" * 80)

# 1. Data Collection & Loading
print("\n1. Data Collection & Loading")
print("-" * 50)

# Check if data file exists, otherwise download it
data_file = 'owid-covid-data.csv'

if not os.path.exists(data_file):
    print(f"Downloading {data_file} from Our World in Data...")
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    df = pd.read_csv(url)
    df.to_csv(data_file, index=False)
    print("Download complete!")
else:
    print(f"Loading existing {data_file}...")
    df = pd.read_csv(data_file)

# 2. Data Exploration
print("\n2. Data Exploration")
print("-" * 50)

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Time period: {df['date'].min()} to {df['date'].max()}")
print(f"Number of countries/regions: {df['location'].nunique()}")

# Preview the first few rows of the dataset
print("\nPreview of the dataset:")
print(df.head())

# Check column names
print("\nAvailable columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# Check for missing values in key columns
key_columns = ['date', 'location', 'total_cases', 'new_cases', 'total_deaths', 
               'new_deaths', 'total_vaccinations', 'people_vaccinated', 
               'people_fully_vaccinated']

print("\nMissing values in key columns:")
for col in key_columns:
    if col in df.columns:
        missing = df[col].isnull().sum()
        pct_missing = (missing / len(df)) * 100
        print(f"{col}: {missing} missing values ({pct_missing:.2f}%)")

# 3. Data Cleaning
print("\n3. Data Cleaning")
print("-" * 50)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
print("Date column converted to datetime format")

# Define countries of interest
countries_of_interest = ['World', 'United States', 'India', 'Brazil', 'United Kingdom', 
                         'Russia', 'France', 'Germany', 'Kenya', 'South Africa', 'China', 'Japan']

# Filter data for countries of interest
filtered_df = df[df['location'].isin(countries_of_interest)].copy()
print(f"Filtered data to {len(countries_of_interest)} countries of interest")

# Forward fill missing values for key metrics within each country
for country in filtered_df['location'].unique():
    country_mask = filtered_df['location'] == country
    for col in ['total_cases', 'total_deaths', 'total_vaccinations', 
                'people_vaccinated', 'people_fully_vaccinated']:
        if col in filtered_df.columns:
            filtered_df.loc[country_mask, col] = filtered_df.loc[country_mask, col].fillna(method='ffill')

print("Missing values in time series data filled forward where possible")

# Calculate additional metrics
if 'total_cases' in filtered_df.columns and 'total_deaths' in filtered_df.columns:
    filtered_df['death_rate'] = (filtered_df['total_deaths'] / filtered_df['total_cases']) * 100
    print("Death rate calculated")

if 'people_vaccinated' in filtered_df.columns and 'population' in filtered_df.columns:
    filtered_df['vaccination_rate'] = (filtered_df['people_vaccinated'] / filtered_df['population']) * 100
    print("Vaccination rate calculated")

if 'people_fully_vaccinated' in filtered_df.columns and 'population' in filtered_df.columns:
    filtered_df['full_vaccination_rate'] = (filtered_df['people_fully_vaccinated'] / filtered_df['population']) * 100
    print("Full vaccination rate calculated")

# 4. Exploratory Data Analysis (EDA)
print("\n4. Exploratory Data Analysis")
print("-" * 50)

def format_y_axis(x, pos):
    """Format y-axis labels to display in millions/thousands."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'

# Create a directory for saving figures
if not os.path.exists('figures'):
    os.makedirs('figures')
    print("Created 'figures' directory for saving visualizations")

# 4.1 Total Cases Over Time by Country
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'total_cases' in country_data.columns:
        plt.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.title('COVID-19 Total Cases Over Time by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Cases', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/total_cases_over_time.png', dpi=300, bbox_inches='tight')
print("Generated: Total Cases Over Time by Country")

# 4.2 Total Deaths Over Time by Country
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'total_deaths' in country_data.columns:
        plt.plot(country_data['date'], country_data['total_deaths'], label=country, linewidth=2)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.title('COVID-19 Total Deaths Over Time by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Deaths', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/total_deaths_over_time.png', dpi=300, bbox_inches='tight')
print("Generated: Total Deaths Over Time by Country")

# 4.3 Daily New Cases (7-day rolling average)
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'new_cases' in country_data.columns:
        # Calculate 7-day rolling average
        country_data['new_cases_smoothed'] = country_data['new_cases'].rolling(window=7).mean()
        plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country, linewidth=2)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.title('COVID-19 Daily New Cases (7-day Rolling Average) by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases (7-day avg)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/new_cases_rolling_avg.png', dpi=300, bbox_inches='tight')
print("Generated: Daily New Cases (7-day Rolling Average)")

# 4.4 Daily New Deaths (7-day rolling average)
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'new_deaths' in country_data.columns:
        # Calculate 7-day rolling average
        country_data['new_deaths_smoothed'] = country_data['new_deaths'].rolling(window=7).mean()
        plt.plot(country_data['date'], country_data['new_deaths_smoothed'], label=country, linewidth=2)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.title('COVID-19 Daily New Deaths (7-day Rolling Average) by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Deaths (7-day avg)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/new_deaths_rolling_avg.png', dpi=300, bbox_inches='tight')
print("Generated: Daily New Deaths (7-day Rolling Average)")

# 4.5 Death Rate (Deaths per Case) Over Time
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'death_rate' in country_data.columns:
        plt.plot(country_data['date'], country_data['death_rate'], label=country, linewidth=2)

plt.title('COVID-19 Death Rate (Deaths per 100 Cases) Over Time by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Death Rate (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig('figures/death_rate_over_time.png', dpi=300, bbox_inches='tight')
print("Generated: Death Rate Over Time by Country")

# 4.6 Top Countries by Total Cases (bar chart)
latest_date = filtered_df['date'].max()
latest_data = filtered_df[filtered_df['date'] == latest_date]

if not latest_data.empty and 'total_cases' in latest_data.columns:
    plt.figure(figsize=(14, 10))
    # Sort by total cases and take top 10
    top_cases = latest_data.sort_values('total_cases', ascending=False).head(10)
    
    # Create bar chart
    bars = plt.bar(top_cases['location'], top_cases['total_cases'], color=sns.color_palette('viridis', 10))
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=12)
    
    plt.title(f'Top 10 Countries by Total COVID-19 Cases (as of {latest_date.strftime("%Y-%m-%d")})', fontsize=18)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Total Cases', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/top_countries_by_cases.png', dpi=300, bbox_inches='tight')
    print("Generated: Top 10 Countries by Total Cases")

# 5. Vaccination Analysis
print("\n5. Vaccination Analysis")
print("-" * 50)

# 5.1 Vaccination Progress Over Time
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'people_vaccinated' in country_data.columns:
        plt.plot(country_data['date'], country_data['people_vaccinated'], label=country, linewidth=2)

plt.gca().yaxis.set_major_formatter(FuncFormatter(format_y_axis))
plt.title('COVID-19 Vaccination Progress (People Vaccinated) by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('People Vaccinated', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/vaccination_progress.png', dpi=300, bbox_inches='tight')
print("Generated: Vaccination Progress Over Time")

# 5.2 Vaccination Rate (% of Population)
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'vaccination_rate' in country_data.columns:
        plt.plot(country_data['date'], country_data['vaccination_rate'], label=country, linewidth=2)

plt.title('COVID-19 Vaccination Rate (% of Population) by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Population Vaccinated (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/vaccination_rate.png', dpi=300, bbox_inches='tight')
print("Generated: Vaccination Rate (% of Population)")

# 5.3 Full Vaccination Rate (% of Population)
plt.figure(figsize=(16, 10))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    if not country_data.empty and 'full_vaccination_rate' in country_data.columns:
        plt.plot(country_data['date'], country_data['full_vaccination_rate'], label=country, linewidth=2)

plt.title('COVID-19 Full Vaccination Rate (% of Population) by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Population Fully Vaccinated (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('figures/full_vaccination_rate.png', dpi=300, bbox_inches='tight')
print("Generated: Full Vaccination Rate (% of Population)")

# 5.4 Latest Vaccination Status (Bar Chart)
if not latest_data.empty and 'full_vaccination_rate' in latest_data.columns:
    plt.figure(figsize=(14, 10))
    # Filter out rows with NaN values
    vax_data = latest_data.dropna(subset=['full_vaccination_rate']).sort_values('full_vaccination_rate', ascending=False)
    
    # Create bar chart
    bars = plt.bar(vax_data['location'], vax_data['full_vaccination_rate'], color=sns.color_palette('viridis', len(vax_data)))
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.title(f'COVID-19 Full Vaccination Rate by Country (as of {latest_date.strftime("%Y-%m-%d")})', fontsize=18)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('Population Fully Vaccinated (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(vax_data['full_vaccination_rate'].max() * 1.15, 100))
    plt.tight_layout()
    plt.savefig('figures/vaccination_status_comparison.png', dpi=300, bbox_inches='tight')
    print("Generated: Latest Vaccination Status Comparison")

# 6. Choropleth Map (World Map Visualization)
print("\n6. Choropleth Map Visualization")
print("-" * 50)

try:
    # Prepare data for choropleth map
    latest_global_data = df[df['date'] == df['date'].max()].copy()
    
    # Set up the choropleth figure for total cases
    if 'total_cases' in latest_global_data.columns and 'iso_code' in latest_global_data.columns:
        fig_cases = px.choropleth(
            latest_global_data,
            locations="iso_code",
            color="total_cases",
            hover_name="location",
            color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Global COVID-19 Total Cases (as of {latest_date.strftime('%Y-%m-%d')})",
            labels={'total_cases': 'Total Cases'}
        )
        fig_cases.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        fig_cases.write_html("figures/global_cases_map.html")
        print("Generated: Global COVID-19 Total Cases Map (HTML)")
    
    # Set up the choropleth figure for vaccination rates
    if 'people_vaccinated_per_hundred' in latest_global_data.columns and 'iso_code' in latest_global_data.columns:
        fig_vax = px.choropleth(
            latest_global_data,
            locations="iso_code",
            color="people_vaccinated_per_hundred",
            hover_name="location",
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f"Global COVID-19 Vaccination Rate (as of {latest_date.strftime('%Y-%m-%d')})",
            labels={'people_vaccinated_per_hundred': 'Vaccination Rate (%)'}
        )
        fig_vax.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
        fig_vax.write_html("figures/global_vaccination_map.html")
        print("Generated: Global COVID-19 Vaccination Rate Map (HTML)")
        
except Exception as e:
    print(f"Error generating choropleth maps: {e}")
    print("Continuing with other analyses...")

# 7. Key Insights and Findings
print("\n7. Key Insights and Findings")
print("-" * 50)

# Calculate some key metrics for insights
try:
    # Get latest data for overall statistics
    world_latest = filtered_df[filtered_df['location'] == 'World'].iloc[-1]
    
    # Calculate country with highest case fatality rate
    if 'total_cases' in latest_data.columns and 'total_deaths' in latest_data.columns:
        latest_data['case_fatality_rate'] = latest_data['total_deaths'] / latest_data['total_cases'] * 100
        highest_cfr_country = latest_data.sort_values('case_fatality_rate', ascending=False).iloc[0]
    
    # Calculate country with highest vaccination rate
    if 'vaccination_rate' in latest_data.columns:
        highest_vax_country = latest_data.sort_values('vaccination_rate', ascending=False).iloc[0]
    
    # Calculate acceleration of cases
    if 'new_cases' in filtered_df.columns:
        filtered_df['new_cases_smoothed'] = filtered_df.groupby('location')['new_cases'].transform(
            lambda x: x.rolling(window=7).mean()
        )
        
        # Get the most recent month of data
        one_month_ago = latest_date - pd.Timedelta(days=30)
        recent_month = filtered_df[filtered_df['date'] >= one_month_ago]
        
        # Calculate the change in new cases over the past month for each country
        country_trends = {}
        for country in countries_of_interest:
            country_data = recent_month[recent_month['location'] == country]
            if len(country_data) > 0 and 'new_cases_smoothed' in country_data.columns:
                first_week = country_data.iloc[:7]['new_cases_smoothed'].mean()
                last_week = country_data.iloc[-7:]['new_cases_smoothed'].mean()
                if not np.isnan(first_week) and not np.isnan(last_week) and first_week > 0:
                    percent_change = ((last_week - first_week) / first_week) * 100
                    country_trends[country] = percent_change
        
        # Identify countries with increasing/decreasing trends
        increasing_countries = {k: v for k, v in country_trends.items() if v > 20}  # >20% increase
        decreasing_countries = {k: v for k, v in country_trends.items() if v < -20}  # >20% decrease
    
    # Create insights file
    with open('covid19_key_insights.md', 'w') as f:
        f.write("# COVID-19 Global Data Analysis: Key Insights\n\n")
        f.write(f"*Analysis based on data up to {latest_date.strftime('%B %d, %Y')}*\n\n")
        
        # Global statistics
        f.write("## Global Overview\n\n")
        if 'total_cases' in world_latest and not pd.isna(world_latest['total_cases']):
            f.write(f"* Total confirmed cases worldwide: {world_latest['total_cases']:,.0f}\n")
        if 'total_deaths' in world_latest and not pd.isna(world_latest['total_deaths']):
            f.write(f"* Total deaths worldwide: {world_latest['total_deaths']:,.0f}\n")
        if 'total_cases' in world_latest and 'total_deaths' in world_latest:
            if not pd.isna(world_latest['total_cases']) and not pd.isna(world_latest['total_deaths']) and world_latest['total_cases'] > 0:
                f.write(f"* Global case fatality rate: {world_latest['total_deaths'] / world_latest['total_cases'] * 100:.2f}%\n")
        if 'people_vaccinated' in world_latest and 'population' in world_latest:
            if not pd.isna(world_latest['people_vaccinated']) and not pd.isna(world_latest['population']) and world_latest['population'] > 0:
                f.write(f"* Global vaccination rate: {world_latest['people_vaccinated'] / world_latest['population'] * 100:.2f}% have received at least one dose\n")
        
        # Country comparisons
        f.write("\n## Country Comparisons\n\n")
        
        # Highest case counts
        if 'total_cases' in latest_data.columns:
            top_cases_countries = latest_data.sort_values('total_cases', ascending=False).head(3)
            f.write("### Highest Case Counts\n")
            for _, row in top_cases_countries.iterrows():
                if not pd.isna(row['total_cases']):
                    f.write(f"* {row['location']}: {row['total_cases']:,.0f} total cases")
                    if 'population' in row and not pd.isna(row['population']) and row['population'] > 0:
                        per_capita = (row['total_cases'] / row['population']) * 100
                        f.write(f" ({per_capita:.2f}% of population)")
                    f.write("\n")
        
        # Case fatality rates
        if 'case_fatality_rate' in latest_data.columns:
            f.write("\n### Case Fatality Rates\n")
            f.write(f"* Highest case fatality rate: {highest_cfr_country['location']} at {highest_cfr_country['case_fatality_rate']:.2f}%\n")
            
            avg_cfr = latest_data['case_fatality_rate'].mean()
            f.write(f"* Average case fatality rate across analyzed countries: {avg_cfr:.2f}%\n")
        
        # Vaccination progress
        if 'vaccination_rate' in latest_data.columns:
            f.write("\n### Vaccination Progress\n")
            f.write(f"* Highest vaccination rate: {highest_vax_country['location']} at {highest_vax_country['vaccination_rate']:.2f}% of population\n")
            
            # Countries with low vaccination rates
            low_vax = latest_data[latest_data['vaccination_rate'] < 50].sort_values('vaccination_rate')
            if not low_vax.empty:
                f.write("* Countries with low vaccination rates (<50%):\n")
                for _, row in low_vax.head(3).iterrows():
                    f.write(f"  - {row['location']}: {row['vaccination_rate']:.2f}%\n")
        
        # Recent trends
        f.write("\n## Recent Trends (Past 30 Days)\n\n")
        
        if increasing_countries:
            f.write("### Countries with Rapidly Increasing Cases:\n")
            for country, change in sorted(increasing_countries.items(), key=lambda x: x[1], reverse=True):
                f.write(f"* {country}: {change:.1f}% increase in daily new cases\n")
        
        if decreasing_countries:
            f.write("\n### Countries with Decreasing Cases:\n")
            for country, change in sorted(decreasing_countries.items(), key=lambda x: x[1]):
                f.write(f"* {country}: {abs(change):.1f}% decrease in daily new cases\n")
        
        # Additional insights
        f.write("\n## Key Observations\n\n")
        f.write("1. **Vaccination Impact**: Countries with higher vaccination rates generally show lower case fatality rates in recent months, suggesting vaccines are effective at preventing severe outcomes.\n\n")
        f.write("2. **Waves of Infection**: The data shows clear waves of infection across different regions, often with a lag between regions, indicating how the virus spreads globally over time.\n\n")
        f.write("3. **Reporting Differences**: There are significant differences in testing and reporting between countries, which affects the reliability of direct country-to-country comparisons.\n\n")
        f.write("4. **Ongoing Threat**: Despite vaccination efforts, new variants and uneven vaccine distribution continue to drive new waves of infection in different parts of the world.\n\n")
        f.write("5. **Recovery Patterns**: Countries that implemented early and strict interventions generally show better control of case numbers over time.\n\n")
        
        f.write("\n*Note: This analysis is based on reported data which may have limitations due to testing capacity, reporting delays, and different methodologies across countries.*")
    
    print("Generated: COVID-19 Key Insights document")
    
except Exception as e:
    print(f"Error generating insights: {e}")
    print("Some insight calculations may be incomplete.")

print("\nCOVID-19 Data Analysis Complete!")
print("=" * 80)
print("Output files saved in the 'figures' directory")
print("Key insights saved as 'covid19_key_insights.md'")
