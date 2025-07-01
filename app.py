import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Helper function to convert names to initials for privacy
def convert_to_initials(name):
    """Convert 'Last, First' format to 'L, F' initials for privacy"""
    if pd.isna(name) or not isinstance(name, str):
        return "N/A"
    
    name = name.strip().upper()
    if ',' in name:
        parts = name.split(',')
        if len(parts) >= 2:
            last_initial = parts[0].strip()[0] if parts[0].strip() else 'X'
            first_initial = parts[1].strip()[0] if parts[1].strip() else 'X'
            return f"{last_initial}, {first_initial}"
    
    words = name.split()
    if len(words) >= 2:
        return f"{words[-1][0]}, {words[0][0]}"
    elif len(words) == 1:
        return f"{words[0][0]}, X"
    
    return "N/A"

# Compensation processing helpers
def format_employment_status(is_fte):
    return "Full-Time Employee" if is_fte else "Part-Time/Hourly"

def create_employee_breakdown(df):
    """Create comprehensive employee breakdown with all compensation metrics"""
    def comp_type(desc):
        desc = str(desc).upper()
        if "REGULAR WAGES" in desc:
            return "base_salary"
        elif "EMPLOYEE BENEFITS" in desc:
            return "benefits"
        else:
            return "other_compensation"
    
    temp = df.copy()
    temp['comp_type'] = temp['description'].apply(comp_type)
    
    breakdown = temp.pivot_table(
        index=['employee_name', 'functional_area', 'is_fte'],
        columns='comp_type', values='net_amount', aggfunc='sum', fill_value=0
    ).reset_index()
    
    for col in ['base_salary', 'benefits', 'other_compensation']:
        if col not in breakdown.columns:
            breakdown[col] = 0.0
    
    breakdown['total_compensation'] = breakdown['base_salary'] + breakdown['benefits'] + breakdown['other_compensation']
    breakdown['employee_initials'] = breakdown['employee_name'].apply(convert_to_initials)
    breakdown['employment_status'] = breakdown['is_fte'].apply(format_employment_status)
    breakdown['benefits_percentage'] = (breakdown['benefits'] / breakdown['total_compensation']).replace([float('inf'), float('nan')], 0) * 100
    
    return breakdown[['employee_name', 'employee_initials', 'functional_area', 'is_fte', 'employment_status',
                     'base_salary', 'benefits', 'other_compensation', 'total_compensation', 'benefits_percentage']]

# Unified formatting helper
def format_currency(value):
    """Format a single currency value"""
    return f"${int(round(value)):,}"

def format_currency_column(df, columns):
    """Format multiple currency columns in a dataframe"""
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(format_currency)
    return df

# Statistics calculation helper
def calculate_group_stats(group, metrics=['base_salary', 'benefits', 'total_compensation']):
    """Calculate comprehensive statistics for a group"""
    if len(group) == 0:
        return {f"{metric}_{stat}": 0 for metric in metrics for stat in ['count', 'sum', 'mean', 'median']}
    
    stats = {}
    for metric in metrics:
        stats[f"{metric}_count"] = len(group)
        stats[f"{metric}_sum"] = group[metric].sum()
        stats[f"{metric}_mean"] = group[metric].mean()
        stats[f"{metric}_median"] = group[metric].median()
    
    return stats

# Data loading functions
@st.cache_data
def get_available_years():
    """Get list of available data years without loading data"""
    data_files = {
        2024: "data/Alpine_School_District_2024.csv",
        2023: "data/Alpine School District_2023.csv", 
        2022: "data/Alpine School District_2022.csv"
    }
    
    available_years = []
    for year, file_path in data_files.items():
        try:
            import os
            if os.path.exists(file_path):
                available_years.append(year)
        except:
            continue
    
    return available_years

@st.cache_data
def load_single_year_data(year):
    """Load data for a single year only"""
    data_files = {
        2024: "data/Alpine_School_District_2024.csv",
        2023: "data/Alpine School District_2023.csv", 
        2022: "data/Alpine School District_2022.csv"
    }
    
    if year not in data_files:
        st.error(f"Data for year {year} not available!")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(data_files[year], 
                        usecols=['employee_name', 'title', 'org2', 'description', 'net_amount'])
        df['fiscal_year'] = year
        return df
    except FileNotFoundError:
        st.error(f"Data file for {year} not found: {data_files[year]}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data for {year}: {str(e)}")
        return pd.DataFrame()

# Classification functions
def classify_functional_area(row):
    """Classify employee into functional area"""
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

def is_fte(row):
    """Determine if employee is full-time equivalent"""
    title = row['title'].lower()
    if any(x in title for x in ["hourly", "substitute", "part-time", "temp"]):
        return False
    return True

# Load and process data
available_years = get_available_years()

# Year selector in sidebar
st.sidebar.markdown("### Data Year Selection")
if available_years:
    selected_year = st.sidebar.selectbox(
        "Select Year for Analysis", 
        available_years, 
        index=0 if 2024 in available_years else 0,
        help="2024 is the most critical year due to pending district split"
    )
    
    df = load_single_year_data(selected_year)
    
    if df.empty:
        st.error(f"Failed to load data for {selected_year}")
        st.stop()
    
    if selected_year == 2024:
        st.sidebar.info("📊 **2024 Data Selected** - Most current year before district reorganization")
else:
    st.error("No data available!")
    st.stop()

# Aspen Peaks school list
aspen_peaks_schools = [
    "American Fork High", "Lone Peak High", "Lehi High", "Willowcreek", "Meadow",
    "Deerfield", "Dry Creek", "Shelley", "River Rock", "Traverse Mountain",
    "Snow Springs", "Freedom", "North Point", "Barratt", "Eaglecrest", "Westfield",
    "Greenwood", "Highland Elementary", "Mountain Ridge", "Legacy", "Cedar Ridge",
    "Alpine Elementary"
]

# Data processing
df = df.dropna(subset=["employee_name", "title", "net_amount"])
df['employee_name'] = df['employee_name'].str.strip().str.upper()
df['functional_area'] = df.apply(classify_functional_area, axis=1)
df['is_fte'] = df.apply(is_fte, axis=1)

# Aspen Peaks filter
st.sidebar.markdown("### Aspen Peaks District Filter")
filter_aspen = st.sidebar.checkbox("Only show Aspen Peaks District")

if filter_aspen:
    df = df[df['org2'].str.contains('|'.join(aspen_peaks_schools), case=False, na=False)]
    df = df[df['functional_area'] != "District Administration"]
    st.info("Aspen Peaks mode enabled — District-level administrators have been excluded from this view.")

# Create employee breakdown
employee_breakdown = create_employee_breakdown(df)

# Sidebar filters
st.sidebar.header("Filters")
selected_types = st.sidebar.multiselect("Filter by Employee Type", employee_breakdown['functional_area'].unique())
filtered_data = employee_breakdown[
    employee_breakdown['functional_area'].isin(selected_types)
] if selected_types else employee_breakdown

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
    - **Classification Methodology**: The grouping of employees into functional areas represents analytical interpretation based on job titles and descriptions
    - **Geographic Boundaries**: The "Aspen Peaks District" filter represents schools serving communities around American Fork, Lehi, Highland, and surrounding areas
    - **Total Compensation**: The figures shown include both salaries and employee benefits (approximately 70% salary, 30% benefits)
    
    ### Author's Note
    The author's wife is included in this dataset.
    """)

st.markdown("---")

# Key Takeaways Section
st.subheader("🎯 Key Takeaways for Community Members")

# Calculate key metrics for all groups
teachers = filtered_data[filtered_data['functional_area'] == "Instruction"]
school_admins = filtered_data[filtered_data['functional_area'] == "School Administration"]
district_admins = filtered_data[filtered_data['functional_area'] == "District Administration"]

teacher_stats = calculate_group_stats(teachers)
school_admin_stats = calculate_group_stats(school_admins)
district_admin_stats = calculate_group_stats(district_admins)

with st.expander("📋 Executive Summary - What the Data Shows", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏫 Teacher Compensation")
        st.metric("Count", teacher_stats['base_salary_count'])
        st.metric("Avg Base Salary", format_currency(teacher_stats['base_salary_mean']))
        st.metric("Avg Benefits", format_currency(teacher_stats['benefits_mean']))
        st.metric("Avg Total", format_currency(teacher_stats['total_compensation_mean']))
    
    with col2:
        st.markdown("### 👥 Administrative Comparison")
        st.metric("School Admin Count", school_admin_stats['base_salary_count'])
        st.metric("School Admin Avg Total", format_currency(school_admin_stats['total_compensation_mean']))
        if len(district_admins) > 0:
            st.metric("District Admin Count", district_admin_stats['base_salary_count'])
            st.metric("District Admin Avg Total", format_currency(district_admin_stats['total_compensation_mean']))

st.markdown("---")

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Employees", filtered_data['employee_name'].nunique())
with col2:
    st.metric("Total Base Salary", format_currency(filtered_data['base_salary'].sum()))
with col3:
    st.metric("Total Benefits", format_currency(filtered_data['benefits'].sum()))
with col4:
    st.metric("Total Compensation", format_currency(filtered_data['total_compensation'].sum()))

# Compensation by Employee Type
st.subheader("Compensation Analysis by Employee Type")

# Bar Chart - Total Compensation by Employee Type (Stacked)
bar_df = filtered_data.groupby("functional_area")[["base_salary", "benefits"]].sum()
fig, ax = plt.subplots(figsize=(10, 5))
bar_df[['base_salary', 'benefits']].plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Compensation ($)")
ax.set_title("Base Salary + Benefits by Functional Area")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Average Compensation Table
avg_comp = filtered_data.groupby("functional_area")[["base_salary", "benefits", "total_compensation"]].mean().sort_values(by="total_compensation", ascending=False)
avg_comp_formatted = format_currency_column(avg_comp, ["base_salary", "benefits", "total_compensation"])
avg_comp_formatted = avg_comp_formatted.rename(columns={
    "base_salary": "Avg Base Salary", 
    "benefits": "Avg Benefits", 
    "total_compensation": "Avg Total Compensation"
})
st.dataframe(avg_comp_formatted)

# Role Comparison Summary
st.subheader("Role Comparison Summary")
comparison_data = {
    'Role': ['Teachers', 'School Administrators', 'District Administrators'],
    'Employee Count': [len(teachers), len(school_admins), len(district_admins)],
    'Avg Total Compensation': [
        format_currency(teacher_stats['total_compensation_mean']),
        format_currency(school_admin_stats['total_compensation_mean']),
        format_currency(district_admin_stats['total_compensation_mean']) if len(district_admins) > 0 else "N/A"
    ],
    'Total Budget': [
        format_currency(teacher_stats['total_compensation_sum']),
        format_currency(school_admin_stats['total_compensation_sum']),
        format_currency(district_admin_stats['total_compensation_sum']) if len(district_admins) > 0 else "N/A"
    ]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# Salary Distribution Analysis
st.subheader("Compensation Distribution by Role")
def comp_stats(df, col):
    return df.groupby('functional_area')[col].agg([
        'count', 'min', 'max', 'mean', 'median', 'std'
    ]).round(0)

stats = comp_stats(filtered_data, 'total_compensation')
stats.columns = ['Count', 'Min', 'Max', 'Mean', 'Median', 'Std Dev']
stats['Range'] = stats['Max'] - stats['Min']
stats = stats.sort_values('Mean', ascending=False)

formatted_stats = format_currency_column(stats, ['Min', 'Max', 'Mean', 'Median', 'Std Dev', 'Range'])
st.dataframe(formatted_stats, use_container_width=True)

# Top Earners Section
st.subheader("Top Earners in District")

sort_by = st.selectbox("Sort by", ["Total Compensation", "Base Salary", "Benefits"], index=0)
top_n = st.selectbox("Show Top", [10, 20, 50], index=1)

sort_col_map = {
    "Total Compensation": "total_compensation",
    "Base Salary": "base_salary", 
    "Benefits": "benefits"
}
sort_col = sort_col_map[sort_by]

top_earners = employee_breakdown.sort_values(by=sort_col, ascending=False).head(top_n).copy()
top_earners_display = top_earners[[
    'employee_initials', 'functional_area', 'employment_status',
    'base_salary', 'benefits', 'total_compensation'
]].rename(columns={
    'employee_initials': 'Employee',
    'functional_area': 'Role', 
    'employment_status': 'Status',
    'base_salary': 'Base Salary',
    'benefits': 'Benefits',
    'total_compensation': 'Total Compensation'
})

top_earners_formatted = format_currency_column(top_earners_display, ['Base Salary', 'Benefits', 'Total Compensation'])
st.dataframe(top_earners_formatted, use_container_width=True)

# Sample job titles
st.subheader("Sample Job Titles by Functional Area")
sample_titles = df.groupby('functional_area')['title'].unique().apply(
    lambda x: ', '.join(list(pd.Series(x).dropna().unique())[:5])
)
st.dataframe(sample_titles.reset_index().rename(columns={'title': 'Sample Titles'}))

# Download section
st.subheader("Download Cleaned Data")
download_data = filtered_data[['employee_initials', 'functional_area', 'is_fte', 'base_salary', 'benefits', 'total_compensation']].copy()
download_data = download_data.rename(columns={'employee_initials': 'employee'})

st.download_button(
    "Download as CSV",
    download_data.to_csv(index=False),
    f"alpine_district_salary_analysis_{selected_year}.csv",
    "text/csv",
    help="CSV file with employee names converted to initials for privacy"
)

# Crosscheck section
st.subheader("Crosscheck with Financial Report")
instruction_percentage = (teachers['total_compensation'].sum() / filtered_data['total_compensation'].sum()) * 100 if filtered_data['total_compensation'].sum() > 0 else 0

st.markdown(f"""
- **Report FTE**: 8,231  
- **Your dataset**: {filtered_data['employee_name'].nunique():,} unique employees
- **Report: 61.9% to Instruction**  
- **App: {instruction_percentage:.1f}% to Teachers**
""")
