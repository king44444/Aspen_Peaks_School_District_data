import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Load data
df = pd.read_csv("data/Alpine_School_District_2024.csv")

# Aspen Peaks school list (based on city boundaries)
aspen_peaks_schools = [
    "American Fork High", "Lone Peak High", "Lehi High", "Willowcreek", "Meadow",
    "Deerfield", "Dry Creek", "Shelley", "River Rock", "Traverse Mountain",
    "Snow Springs", "Freedom", "North Point", "Barratt", "Eaglecrest", "Westfield",
    "Greenwood", "Highland Elementary", "Mountain Ridge", "Legacy", "Cedar Ridge",
    "Alpine Elementary"
]

# Basic cleanup
df = df.dropna(subset=["employee_name", "title", "net_amount"])
df['employee_name'] = df['employee_name'].str.strip().str.upper()  # Normalize names

# Aspen Peaks filter
st.sidebar.markdown("### Aspen Peaks District Filter")
filter_aspen = st.sidebar.checkbox("Only show Aspen Peaks District")

if filter_aspen:
    df = df[df['org2'].str.contains('|'.join(aspen_peaks_schools), case=False, na=False)]

# Classify employee type based on title and description
def classify(row):
    title = row['title'].lower()
    desc = str(row['description']).lower()

    if "teacher" in title or "cert" in title:
        return "Instruction"
    elif "principal" in title or "school admin" in desc:
        return "School Administration"
    elif any(x in title for x in ["superintendent", "director", "district admin"]) or "board" in desc:
        return "District Administration"
    elif any(x in title for x in ["counselor", "psychologist", "speech"]):
        return "Student Support"
    elif any(x in title for x in ["custodian", "maintenance", "facilities"]):
        return "Operations & Maintenance"
    elif "transport" in title:
        return "Transportation"
    elif "food" in title or "nutrition" in title:
        return "Food Services"
    else:
        return "Other"

df['functional_area'] = df.apply(classify, axis=1)

# Apply district administration exclusion for Aspen Peaks mode
if filter_aspen:
    df = df[df['functional_area'] != "District Administration"]
    st.info("Aspen Peaks mode enabled — District-level administrators have been excluded from this view.")

def is_fte(row):
    title = row['title'].lower()
    if any(x in title for x in ["hourly", "substitute", "part-time", "temp"]):
        return False
    return True

df['is_fte'] = df.apply(is_fte, axis=1)

# Aggregate total salary per employee
employee_totals = (
    df.groupby(['employee_name', 'functional_area', 'is_fte'])['net_amount']
    .sum()
    .reset_index()
)

# Sidebar filters
st.sidebar.header("Filters")
selected_types = st.sidebar.multiselect("Filter by Employee Type", employee_totals['functional_area'].unique())

filtered_data = employee_totals[
    employee_totals['functional_area'].isin(selected_types)
] if selected_types else employee_totals

# Format helper
def format_currency_column(df, column='net_amount'):
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].apply(lambda x: f"${int(round(x)):,.0f}")
    return df

# Pie chart helper
def small_pie_chart(values, labels, colors):
    fig, ax = plt.subplots(figsize=(2, 2))
    
    # Validate input data
    if not values or len(values) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Check for NaN or invalid values
    import numpy as np
    clean_values = []
    clean_colors = []
    clean_labels = []
    
    for i, val in enumerate(values):
        if not (np.isnan(val) or val <= 0):
            clean_values.append(val)
            if i < len(colors):
                clean_colors.append(colors[i])
            if i < len(labels):
                clean_labels.append(labels[i])
    
    # If no valid data after cleaning, show "No Data"
    if not clean_values:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig
    
    # Create pie chart with clean data
    ax.pie(clean_values, labels=None, colors=clean_colors, startangle=90, autopct='%1.1f%%', textprops={'fontsize': 7})
    ax.axis('equal')
    return fig

# Title and Attribution
st.title("Alpine School District - Salary Analysis")
st.markdown("*Analysis by Michael King*")

# Data Source and Disclaimer
with st.expander("📊 Data Source & Important Disclaimers", expanded=False):
    st.markdown("""
    ### Data Source
    All salary data comes directly from **Utah's official transparency portal** at [transparent.utah.gov](https://transparent.utah.gov). 
    The underlying financial information represents official state records of public employee compensation.
    
    ### Important Disclaimers
    - **Data Accuracy**: The raw salary amounts are official state data and are accurate as reported
    - **Classification Methodology**: The grouping of employees into functional areas (Instruction, Administration, etc.) represents analytical interpretation based on job titles and descriptions
    - **Interpretive Elements**: While the classification system is systematic and logical, it may not perfectly align with the district's internal organizational structure
    - **Geographic Boundaries**: The "Aspen Peaks District" filter represents schools serving communities around American Fork, Lehi, Highland, and surrounding areas within Alpine School District
    - **District-level Employees Excluded**: When the Aspen Peaks filter is enabled, district-level administrators are excluded from analysis. This reflects local school board priorities and enables clearer analysis of site-based staffing and compensation.
    
    ### Best Practices for Interpretation
    - Focus on salary ranges and medians rather than individual outliers
    - Consider these groupings as analytical tools, not official district classifications
    - Cross-reference findings with official district budget documents when possible
    - **Total Compensation**: The figures shown include both salaries and employee benefits (approximately 70% salary, 30% benefits)
    
    ### Disclaimer
    This tool is still in ALPHA. Data structures, formatting, and features may change. Expect updates and occasional bugs as we improve the experience.
    
    ### Author's Note
    The author's wife is included in this dataset.
    """)
    
    # # Search for and display author's wife record (NIKKI KING at Freedom Elementary)
    # wife_records = df[df['employee_name'].str.contains('KING, NIKKI', case=False, na=False)]
    
    # if not wife_records.empty:
    #     # Get the aggregated salary data for NIKKI KING
    #     wife_name = "KING, NIKKI"
    #     wife_total = df[df['employee_name'] == wife_name]['net_amount'].sum()
    #     wife_functional_area = "Instruction"  # She's primarily a teacher
    #     wife_is_fte = True  # She has full-time certified teacher salary
        
    #     # Create a display record matching the target schema format
    #     wife_display = pd.DataFrame({
    #         'employee_name': [wife_name],
    #         'functional_area': [wife_functional_area],
    #         'is_fte': [wife_is_fte],
    #         'net_amount': [wife_total]
    #     })
        
    #     st.markdown("**Author's Wife - Freedom Elementary SPED Teacher:**")
    #     st.dataframe(format_currency_column(wife_display), use_container_width=True)
        
    #     # Show breakdown of her compensation (simple display without nested expander)
    #     st.markdown("**Compensation Breakdown:**")
    #     wife_breakdown = df[df['employee_name'] == wife_name][['org2', 'title', 'net_amount']].copy()
    #     wife_breakdown['net_amount'] = wife_breakdown['net_amount'].apply(lambda x: f"${x:,.2f}")
    #     st.dataframe(wife_breakdown, use_container_width=True)
    # else:
    #     st.markdown("*No matching record found for Freedom Elementary SPED position.*")

st.markdown("---")

# Main metrics
st.metric("Total Unique Employees", filtered_data['employee_name'].nunique())
st.metric("Total Salary (All Time)", f"${int(round(filtered_data['net_amount'].sum())):,}")

# Bar Chart - Total Salary per Employee Type
st.subheader("Total Salary by Employee Type")
st.bar_chart(
    filtered_data.groupby("functional_area")["net_amount"].sum()
)

# Average Salary per Employee Type
st.subheader("Average Salary per Employee Type")
avg_salary = filtered_data.groupby("functional_area")["net_amount"].mean().sort_values(ascending=False)
st.dataframe(avg_salary.apply(lambda x: f"${int(round(x)):,}"))

# Pie chart - Proportion of Budget per Role Type
st.subheader("Salary Distribution by Employee Type")
fig = plt.figure(figsize=(6,6))
plt.pie(list(avg_salary.values), labels=list(avg_salary.index), autopct="%1.1f%%", startangle=90)
plt.title("Budget Share by Role")
plt.axis('equal')
st.pyplot(fig)

# === Teachers vs School Administrators Comparison ===
st.subheader("Teachers vs School Administrators")
st.markdown("*Comparing teachers with principals and school-level administrators*")

teachers = filtered_data[filtered_data['functional_area'] == "Instruction"]
school_admins = filtered_data[filtered_data['functional_area'] == "School Administration"]

employee_counts_school = [len(teachers), len(school_admins)]
salary_sums_school = [teachers['net_amount'].sum(), school_admins['net_amount'].sum()]
avg_salaries_school = [teachers['net_amount'].mean(), school_admins['net_amount'].mean()]
labels_school = ['Teachers', 'School Admins']
colors = ['#4e79a7', '#f28e2c']

# Row 1: Employee Count
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Teachers", f"{employee_counts_school[0]} employees")
with col2:
    st.pyplot(small_pie_chart(employee_counts_school, labels_school, colors))
with col3:
    st.metric("School Administrators", f"{employee_counts_school[1]} employees")

# Row 2: Total Salary
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Total Teacher Salary", f"${int(round(salary_sums_school[0])):,}")
with col2:
    st.pyplot(small_pie_chart(salary_sums_school, labels_school, colors))
with col3:
    st.metric("Total School Admin Salary", f"${int(round(salary_sums_school[1])):,}")

# Row 3: Average Salary
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Avg Teacher Salary", f"${int(round(avg_salaries_school[0])):,}")
with col2:
    st.pyplot(small_pie_chart(avg_salaries_school, labels_school, colors))
with col3:
    st.metric("Avg School Admin Salary", f"${int(round(avg_salaries_school[1])):,}")

# === Teachers vs District Administrators Comparison ===
st.subheader("Teachers vs District Administrators")
st.markdown("*Comparing teachers with superintendents, directors, and district-level staff*")

district_admins = filtered_data[filtered_data['functional_area'] == "District Administration"]

employee_counts_district = [len(teachers), len(district_admins)]
salary_sums_district = [teachers['net_amount'].sum(), district_admins['net_amount'].sum()]
avg_salaries_district = [teachers['net_amount'].mean(), district_admins['net_amount'].mean()]
labels_district = ['Teachers', 'District Admins']
colors_district = ['#4e79a7', '#e15759']

# Row 1: Employee Count
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Teachers", f"{employee_counts_district[0]} employees")
with col2:
    st.pyplot(small_pie_chart(employee_counts_district, labels_district, colors_district))
with col3:
    st.metric("District Administrators", f"{employee_counts_district[1]} employees")

# Row 2: Total Salary
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Total Teacher Salary", f"${int(round(salary_sums_district[0])):,}")
with col2:
    st.pyplot(small_pie_chart(salary_sums_district, labels_district, colors_district))
with col3:
    st.metric("Total District Admin Salary", f"${int(round(salary_sums_district[1])):,}")

# Row 3: Average Salary
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col1:
    st.metric("Avg Teacher Salary", f"${int(round(avg_salaries_district[0])):,}")
with col2:
    st.pyplot(small_pie_chart(avg_salaries_district, labels_district, colors_district))
with col3:
    if pd.isna(avg_salaries_district[1]):
        st.metric("Avg District Admin Salary", "N/A (excluded from Aspen Peaks view)")
    else:
        st.metric("Avg District Admin Salary", f"${int(round(avg_salaries_district[1])):,}")

# Summary comparison table
st.markdown("#### Summary: Teacher vs Administrator Compensation")

# Calculate district admin values with null safety
district_avg_salary = "N/A (excluded)" if len(district_admins) == 0 or pd.isna(district_admins['net_amount'].mean()) else f"${int(round(district_admins['net_amount'].mean())):,}"
district_total_budget = "N/A (excluded)" if len(district_admins) == 0 else f"${int(round(district_admins['net_amount'].sum())):,}"

comparison_data = {
    'Role': ['Teachers', 'School Administrators', 'District Administrators'],
    'Employee Count': [len(teachers), len(school_admins), len(district_admins)],
    'Average Salary': [f"${int(round(teachers['net_amount'].mean())):,}", 
                      f"${int(round(school_admins['net_amount'].mean())):,}", 
                      district_avg_salary],
    'Total Budget': [f"${int(round(teachers['net_amount'].sum())):,}", 
                    f"${int(round(school_admins['net_amount'].sum())):,}", 
                    district_total_budget]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# FTE vs Non-FTE Summary
st.subheader("FTE vs Non-FTE Comparison")
fte_data = filtered_data[filtered_data['is_fte']]
non_fte_data = filtered_data[~filtered_data['is_fte']]
col1, col2 = st.columns(2)
with col1:
    st.metric("FTE Count", fte_data['employee_name'].nunique())
    st.metric("FTE Total Salary", f"${int(round(fte_data['net_amount'].sum())):,}")
with col2:
    st.metric("Non-FTE Count", non_fte_data['employee_name'].nunique())
    st.metric("Non-FTE Total Salary", f"${int(round(non_fte_data['net_amount'].sum())):,}")

# === Salary Distribution Analysis ===
st.subheader("Salary Distribution by Role")
st.markdown("This section shows how salaries are distributed within each functional area, helping identify salary ranges and equity within roles.")

# Calculate salary statistics by functional area
salary_stats = filtered_data.groupby('functional_area')['net_amount'].agg([
    'count', 'min', 'max', 'mean', 'median', 
    lambda x: x.quantile(0.25),  # Q1
    lambda x: x.quantile(0.75),  # Q3
    'std'
]).round(0)

salary_stats.columns = ['Employee Count', 'Min Salary', 'Max Salary', 'Mean Salary', 'Median Salary', 'Q1 (25%)', 'Q3 (75%)', 'Std Dev']

# Add salary range calculation
salary_stats['Salary Range'] = salary_stats['Max Salary'] - salary_stats['Min Salary']

# Sort by mean salary for better readability
salary_stats = salary_stats.sort_values('Mean Salary', ascending=False)

# Display formatted statistics table
st.markdown("#### Salary Statistics by Functional Area")
formatted_stats = salary_stats.copy()
currency_cols = ['Min Salary', 'Max Salary', 'Mean Salary', 'Median Salary', 'Q1 (25%)', 'Q3 (75%)', 'Salary Range', 'Std Dev']
for col in currency_cols:
    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"${int(x):,}")

st.dataframe(formatted_stats, use_container_width=True)

# Box Plot - Salary Distribution by Role
st.markdown("#### Salary Distribution Box Plot")
st.markdown("Box plots show the salary spread within each role. The box shows the middle 50% of salaries, with the line inside showing the median.")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for box plot
plot_data = []
plot_labels = []
for area in salary_stats.index:
    role_data = filtered_data[filtered_data['functional_area'] == area]['net_amount']
    if len(role_data) > 0:
        plot_data.append(role_data)
        plot_labels.append(f"{area}\n(n={len(role_data)})")

# Create box plot
box_plot = ax.boxplot(plot_data, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2))
ax.set_xticklabels(plot_labels)

ax.set_title('Salary Distribution by Functional Area', fontsize=14, fontweight='bold')
ax.set_ylabel('Salary ($)', fontsize=12)
ax.set_xlabel('Functional Area', fontsize=12)

# Format y-axis as currency
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(fig)

# Salary Range Comparison
st.markdown("#### Salary Range Analysis")
col1, col2 = st.columns(2)

with col1:
    # Bar chart showing salary ranges
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = range(len(salary_stats))
    bars = ax.barh(y_pos, salary_stats['Salary Range'], color='skyblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(salary_stats.index)
    ax.set_xlabel('Salary Range ($)')
    ax.set_title('Salary Range by Functional Area')
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'${width:,.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Employee count vs Average salary scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(salary_stats['Employee Count'], salary_stats['Mean Salary'], 
                        s=100, alpha=0.7, c='coral')
    
    # Add labels for each point
    for i, area in enumerate(salary_stats.index):
        try:
            x_val = int(pd.to_numeric(salary_stats.loc[area, 'Employee Count']))
            y_val = float(pd.to_numeric(salary_stats.loc[area, 'Mean Salary']))
            ax.annotate(area, (x_val, y_val), xytext=(5, 5), textcoords='offset points', fontsize=9)
        except (ValueError, TypeError):
            # Skip annotation if conversion fails
            continue
    
    ax.set_xlabel('Number of Employees')
    ax.set_ylabel('Average Salary ($)')
    ax.set_title('Employee Count vs Average Salary by Role')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    st.pyplot(fig)

# Salary Distribution Histogram
st.markdown("#### Salary Distribution Histogram by Role")
st.markdown("This histogram shows the frequency of different salary levels within each functional area.")

# Create subplot for each functional area with significant employee count
areas_to_plot = salary_stats[salary_stats['Employee Count'] >= 3].index[:6]  # Top 6 areas with at least 3 employees

if len(areas_to_plot) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum', 'khaki', 'lightgray']
    
    for i, area in enumerate(areas_to_plot):
        if i < len(axes):
            role_salaries = filtered_data[filtered_data['functional_area'] == area]['net_amount']
            
            axes[i].hist(role_salaries, bins=min(10, len(role_salaries)//2 + 1), 
                        color=colors[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{area}\n({len(role_salaries)} employees)', fontsize=10)
            axes[i].set_xlabel('Salary ($)')
            axes[i].set_ylabel('Count')
            
            # Format x-axis as currency
            axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            # Add mean line
            mean_sal = role_salaries.mean()
            axes[i].axvline(mean_sal, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: ${mean_sal:,.0f}')
            axes[i].legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(len(areas_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Not enough data for histogram visualization. Need at least 3 employees per role.")


# Sample job titles for each role
st.subheader("Sample Job Titles by Functional Area")
sample_titles = df.groupby('functional_area')['title'].unique().apply(lambda x: ', '.join(list(pd.Series(x).dropna().unique())[:5]))
st.dataframe(sample_titles.reset_index().rename(columns={'title': 'Sample Titles'}))

# Top earners table
st.subheader("Top 20 Earners in District")
top_earners = filtered_data.sort_values(by="net_amount", ascending=False).head(20)
st.dataframe(format_currency_column(top_earners))

# Raw data download (CSV keeps raw numbers, not formatted strings)
st.subheader("Download Cleaned Data")
st.download_button("Download as CSV", filtered_data.to_csv(index=False), "cleaned_salary_data.csv", "text/csv")

st.subheader("Crosscheck with Financial Report")
st.markdown("""
- **Report FTE**: 8,231  
- **Your dataset names**: 14,897 (includes duplicates, subs, part-time)  
- **Report: 61.9% to Instruction**  
- **App: {}% to Teachers**  
""".format(
    round((teachers['net_amount'].sum() / filtered_data['net_amount'].sum()) * 100)
))
