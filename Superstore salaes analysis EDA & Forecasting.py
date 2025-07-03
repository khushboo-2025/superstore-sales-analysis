#!/usr/bin/env python
# coding: utf-8

# # Dataset Introduction¶
# This dataset contains 9,800 retail transactions from a global superstore spanning across four years. It captures detailed information about each order, including:
# 
# Dates: Order and shipping dates
# 
# Customer Info: Customer ID, name, and segment
# 
# Geography: Country, city, state, postal code, and region
# 
# Product Info: Product ID, category, sub-category, and name
# 
# Order Details: Sales value, quantity, discount, and profit
# 
# The dataset is clean and well-structured, with only a few missing postal codes, making it ideal for robust business analytics and forecasting.
# 
# 

# In[1]:


# Step 1: Import necessary libraries
import pandas as pd

# Step 2: Load the dataset
file_path = "D:\Excel\Superstore Dataset.csv"
df = pd.read_csv(file_path)


# In[2]:


# Step 3: Show basic information
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
display(df.head())


# In[3]:


# Step 4: Overview of column data types and missing values
print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# # Data Cleaning & Preprocessing

# In[4]:


# Convert Order Date and Ship Date to datetime with dayfirst=True
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Drop 'Row ID' – not meaningful
df.drop(columns=['Row ID'], inplace=True)

# Check missing Postal Code entries
print("Missing postal codes:")
print(df[df['Postal Code'].isnull()][['City', 'State']].drop_duplicates())


# In[5]:


# Desired ZIP code to use for Burlington, Vermont
default_burlington_zip = 5401  # We search the internet for the Postal Code

# Impute missing postal codes specifically for Burlington, VT
mask = df['Postal Code'].isnull() &        (df['City'] == 'Burlington') &        (df['State'] == 'Vermont')
df.loc[mask, 'Postal Code'] = default_burlington_zip

# Recheck missing values across dataset
print("\nRemaining missing values after ZIP imputation:")
print(df.isnull().sum())


# # EDA Yearly Sales Trend

# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

#Extract year from order date 
df['Year'] = df['Order Date'].dt.year

#Aggregate sales by year

yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()

#plot
plt.figure(figsize=(10,5))
sns.barplot(data = yearly_sales, x='Year',y='Sales',palette ='viridis')

#label and title

plt.title("Total Sales by Year", fontsize = 16)
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


# Sales increased significantly over time after 2016.
# 
# 2015 -> 2016 ~ -4%
# 
# 2016 -> 2017 ~ +30%
# 
# 2017 -> 2018 ~ +20%
# 
# This shows accelerating bisiness performance likely due to:
#   1. Better Marketing and promotions
#   2. High performing product categories
#   3. Expanded Customer base

# # Regional Sales Analysis

# In[7]:


#group sales by region
regional_sales = df.groupby('Region')['Sales'].sum().reset_index().sort_values(by ='Sales', ascending = False)

#plot
plt.figure(figsize=(10,5))
sns.barplot(data = regional_sales, x= 'Region', y = 'Sales', palette = 'magma')

#labels and title
plt.title("Total Sales by Region", fontsize= 16)
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()


# West Region leads with highest total sales of 
# approaximately \$700,000, followed by east.
# Central and south lag significantally, south being the lowest sales approaximately $390,000.
# 
# West and East are the core revenue drivers, suggesting higher market maturity or stronger customer bases. 
# 
# Central and South could represent growth opportunities with targeted marketing or logistics improvements.
# 
# 

# # Category and Sub Category Sales Analysis

# In[8]:


category_sales = df.groupby('Category')['Sales'].sum().reset_index().sort_values(by='Sales',ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data =category_sales,x='Category',y='Sales', palette ='Set2')
plt.title("Total Sales by Category", fontsize = 16)
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# Technology being the highest revenue generator (above 800,000 dollars)
# followed by furniture and office suplies being nearly together.
# 
# Technology is most profitable or in demand segment.
# 

# In[9]:


sub_category_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index().sort_values(by='Sales',ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data =sub_category_sales,x='Sub-Category',y='Sales', palette ='Set2')
plt.title("Total Sales by Sub-Category", fontsize = 16)
plt.xlabel("Sub-Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# phones and chairs dominate each generating over 300,000 dollars in sales, followed by storage, tables, binders being stronh secondary contributors.
# 
# Items like Machines, Accessories, and Copiers also contribute meaningfully.
# 
# Art,Envelopes, Labels, and Fasteners have very low sales, likely niche or low-demand products.
# 
# Focus on expanding inventory and promotions around high-performing sub-categories. Consider evaluating pricing, marketing, or even phasing out low-sales items if margins are poor.

# # Customer Segment Analysis

# In[10]:


segment_sales = df.groupby('Segment')['Sales'].sum().reset_index().sort_values(by='Sales', ascending = False)

plt.figure(figsize=(8,5))
sns.barplot(data=segment_sales, x='Segment', y ='Sales', palette = 'pastel')

plt.title("Total Sales by Customer Segment", fontsize =16)
plt.xlabel("Segment")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# Consumer are the dominant revenue source, contributing over 1.1 million dollar.
# corporate segment follows along with strong performance of 670k dollar
# Home Office accounts for the smallest share, around 430K dollars.
# 
# 

# # Forcasting sales for the next 7 days using Prophet Model

# In[11]:


pip install prophet


# In[13]:


daily_sales =  df.groupby('Order Date')['Sales'].sum().reset_index()

daily_sales.rename(columns={'Order Date':'ds','Sales':'y'},inplace = True)
daily_sales.head()


# In[14]:


from prophet import Prophet


# Initialize the model
model = Prophet()

# Fit the model on your data
model.fit(daily_sales)


# In[17]:


#create future dataframe
future = model.make_future_dataframe(periods=7)

# forecast
forecast = model.predict(future)


# In[18]:


# Forecast plot
model.plot(forecast)
plt.title("Sales Forecast for Next 7 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()


# In[21]:


# Get last date in original dataset
last_date = daily_sales['ds'].max()

# Filter only the next 7 days
future_forecast = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

# Rename columns for clarity
future_forecast.rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Sales',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
}, inplace=True)

# Calculate average predicted sales
avg_sales = future_forecast['Predicted Sales'].mean()

# Create a new row for the average
average_row = pd.DataFrame({
    'Date': ['Average'],
    'Predicted Sales': [round(avg_sales, 2)],
    'Lower Bound': [None],
    'Upper Bound': [None]
})

# Append the row to the forecast DataFrame
forecast_with_avg = pd.concat([future_forecast, average_row], ignore_index=True)

# Display
print("📅 7-Day Sales Forecast with Average:")
display(forecast_with_avg)


# ### Prophet Model Forecast
# 
# The Prophet model predicts an average of ~2,008 units/day over the next 7 days. The forecast indicates stable sales without extreme volatility, guiding moderate restocking and staffing needs.

# In[ ]:




