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
    
    # Handle different name formats
    name = name.strip().upper()
    if ',' in name:
        parts = name.split(',')
        if len(parts) >= 2:
            last_initial = parts[0].strip()[0] if parts[0].strip() else 'X'
            first_initial = parts[1].strip()[0] if parts[1].strip() else 'X'
            return f"{last_initial}, {first_initial}"
    
    # Fallback for unusual formats
    words = name.split()
    if len(words) >= 2:
        return f"{words[-1][0]}, {words[0][0]}"
    elif len(words) == 1:
        return f"{words[0][0]}, X"
    
    return "N/A"

# --- New: Salary/Benefits breakdown processing and formatting helpers ---
def format_employment_status(is_fte):
    return "Full-Time Employee" if is_fte else "Part-Time/Hourly"

def create_employee_breakdown(df):
    # Classify compensation type from description
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
    # Pivot/aggregate per employee
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
    # Reorder columns for clarity
    breakdown = breakdown[['employee_name', 'employee_initials', 'functional_area', 'is_fte', 'employment_status',
                           'base_salary', 'benefits', 'other_compensation', 'total_compensation', 'benefits_percentage']]
    return breakdown
# --- End new helper block ---

# Optimized data loading - load only available years list first
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
            # Just check if file exists without loading
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
        # Load only essential columns to reduce memory
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

# Get available years without loading all data
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
    
    # Load data for selected year only
    df = load_single_year_data(selected_year)
    
    if df.empty:
        st.error(f"Failed to load data for {selected_year}")
        st.stop()
    
    if selected_year == 2024:
        st.sidebar.info("📊 **2024 Data Selected** - Most current year before district reorganization")
else:
    st.error("No data available!")
    st.stop()

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

# --- New: Use create_employee_breakdown for aggregation ---
employee_breakdown = create_employee_breakdown(df)
# --- End aggregation block ---

# Sidebar filters
st.sidebar.header("Filters")
selected_types = st.sidebar.multiselect("Filter by Employee Type", employee_breakdown['functional_area'].unique())

filtered_data = employee_breakdown[
    employee_breakdown['functional_area'].isin(selected_types)
] if selected_types else employee_breakdown

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

# Key Takeaways Section - Community Focus
st.subheader("🎯 Key Takeaways for Community Members")
st.markdown("*Critical insights from the salary data analysis*")

with st.expander("📋 Executive Summary - What the Data Shows", expanded=True):
    # Calculate key metrics for takeaways (dual-metric version)
    teachers = filtered_data[filtered_data['functional_area'] == "Instruction"]
    school_admins = filtered_data[filtered_data['functional_area'] == "School Administration"]
    district_admins = filtered_data[filtered_data['functional_area'] == "District Administration"]

    # Dual-metric calculations
    def dual_stats(group):
        if len(group) == 0:
            return 0, 0, 0, 0, 0, 0
        avg_salary = group['base_salary'].mean()
        avg_benefits = group['benefits'].mean()
        avg_total = group['total_compensation'].mean()
        median_salary = group['base_salary'].median()
        median_benefits = group['benefits'].median()
        median_total = group['total_compensation'].median()
        return avg_salary, avg_benefits, avg_total, median_salary, median_benefits, median_total

    t_avg_sal, t_avg_ben, t_avg_tot, t_med_sal, t_med_ben, t_med_tot = dual_stats(teachers)
    sa_avg_sal, sa_avg_ben, sa_avg_tot, sa_med_sal, sa_med_ben, sa_med_tot = dual_stats(school_admins)
    da_avg_sal, da_avg_ben, da_avg_tot, da_med_sal, da_med_ben, da_med_tot = dual_stats(district_admins)

    # Range for teachers
    teacher_salary_range = teachers['base_salary'].max() - teachers['base_salary'].min() if len(teachers) > 0 else 0
    teacher_benefit_range = teachers['benefits'].max() - teachers['benefits'].min() if len(teachers) > 0 else 0
    teacher_total_range = teachers['total_compensation'].max() - teachers['total_compensation'].min() if len(teachers) > 0 else 0

    # Budget: sum by base, benefits, total
    total_budget_salary = filtered_data['base_salary'].sum()
    total_budget_benefits = filtered_data['benefits'].sum()
    total_budget_total = filtered_data['total_compensation'].sum()
    instruction_salary = teachers['base_salary'].sum()
    instruction_benefits = teachers['benefits'].sum()
    instruction_total = teachers['total_compensation'].sum()
    instruction_percentage_salary = (instruction_salary / total_budget_salary * 100) if total_budget_salary > 0 else 0
    instruction_percentage_total = (instruction_total / total_budget_total * 100) if total_budget_total > 0 else 0

    st.markdown(f"""
    ### 🏫 Teacher Compensation Reality ({selected_year})

    **💰 Average Teacher Compensation**
    - **Base Salary:** ${t_avg_sal:,.0f}
    - **Benefits:** ${t_avg_ben:,.0f}
    - **Total Compensation:** ${t_avg_tot:,.0f}

    **Median Teacher Compensation**
    - **Base Salary:** ${t_med_sal:,.0f}
    - **Benefits:** ${t_med_ben:,.0f}
    - **Total:** ${t_med_tot:,.0f}

    **Teacher Pay Range**
    - **Base Salary:** ${teacher_salary_range:,.0f}
    - **Benefits:** ${teacher_benefit_range:,.0f}
    - **Total:** ${teacher_total_range:,.0f}

    ### 📊 Budget Allocation Priorities
    - **{instruction_percentage_salary:.1f}% of base salary** goes to classroom instruction
    - **{instruction_percentage_total:.1f}% of total compensation** goes to classroom instruction
    - **{len(teachers):,} teachers** serve the entire district
    - Teachers represent the **largest employee group** but have **administrative oversight** at multiple levels
    """)

    st.markdown(f"""
    ### ⚖️ Administrative vs. Teacher Comparison
    - **Average School Admin (Base/Benefits/Total):** ${sa_avg_sal:,.0f} / ${sa_avg_ben:,.0f} / ${sa_avg_tot:,.0f}
    - **Average Teacher (Base/Benefits/Total):** ${t_avg_sal:,.0f} / ${t_avg_ben:,.0f} / ${t_avg_tot:,.0f}
    """)

    if len(district_admins) > 0 and not filter_aspen:
        st.markdown(f"""
        - **Average District Admin (Base/Benefits/Total):** ${da_avg_sal:,.0f} / ${da_avg_ben:,.0f} / ${da_avg_tot:,.0f}
        """)

    st.markdown(f"""
    **Community Considerations:**
    - Both base salary and benefits are important in evaluating competitiveness
    - Administrative overhead and benefits represent a significant part of the total budget
    - Pay equity varies by both salary and benefits components

    *This analysis reflects {selected_year} data, which is critical as the district undergoes reorganization.*
    """)

st.markdown("---")

# Main metrics
st.metric("Total Unique Employees", filtered_data['employee_name'].nunique())
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Base Salary", f"${int(round(filtered_data['base_salary'].sum())):,}")
with col2:
    st.metric("Total Benefits", f"${int(round(filtered_data['benefits'].sum())):,}")
with col3:
    st.metric("Total Compensation", f"${int(round(filtered_data['total_compensation'].sum())):,}")

# Bar Chart - Total Compensation by Employee Type (Stacked)
st.subheader("Total Compensation by Employee Type")
bar_df = filtered_data.groupby("functional_area")[["base_salary", "benefits"]].sum()

# Create a more memory-efficient plot
fig, ax = plt.subplots(figsize=(10, 5))
bar_df[['base_salary', 'benefits']].plot(kind="bar", stacked=True, ax=ax)
ax.set_ylabel("Compensation ($)")
ax.set_title("Base Salary + Benefits by Functional Area")
plt.tight_layout()
st.pyplot(fig)

# Explicit memory cleanup
plt.close(fig)
del bar_df, fig, ax

# Average Compensation by Employee Type (all three metrics)
st.subheader("Average Compensation by Employee Type")
avg_comp = filtered_data.groupby("functional_area")[["base_salary", "benefits", "total_compensation"]].mean().sort_values(by="total_compensation", ascending=False)
for col in ['base_salary', 'benefits', 'total_compensation']:
    avg_comp[col] = avg_comp[col].apply(lambda x: f"${int(round(x)):,}")
avg_comp = avg_comp.rename(columns={"base_salary": "Avg Base Salary", "benefits": "Avg Benefits", "total_compensation": "Avg Total Compensation"})
st.dataframe(avg_comp)

# District-wide Salary vs Benefits Breakdown
st.subheader("District-wide Salary vs Benefits Breakdown")
total_base = filtered_data['base_salary'].sum()
total_benefits = filtered_data['benefits'].sum()
total_comp = total_base + total_benefits

base_percentage = (total_base / total_comp) * 100 if total_comp > 0 else 0
benefits_percentage = (total_benefits / total_comp) * 100 if total_comp > 0 else 0

col1, col2 = st.columns(2)
with col1:
    st.metric("Base Salary", f"${total_base:,.0f}", f"{base_percentage:.1f}% of total")
with col2:
    st.metric("Benefits", f"${total_benefits:,.0f}", f"{benefits_percentage:.1f}% of total")

# Memory cleanup
del total_base, total_benefits, total_comp, base_percentage, benefits_percentage

# === Teachers vs School Administrators Comparison (all three metrics) ===
st.subheader("Teachers vs School Administrators")
st.markdown("*Comparing teachers with principals and school-level administrators*")

teachers = filtered_data[filtered_data['functional_area'] == "Instruction"]
school_admins = filtered_data[filtered_data['functional_area'] == "School Administration"]
t = teachers
sa = school_admins

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Teachers: Base", f"${int(round(t['base_salary'].sum())):,}")
    st.metric("Teachers: Benefits", f"${int(round(t['benefits'].sum())):,}")
    st.metric("Teachers: Total", f"${int(round(t['total_compensation'].sum())):,}")
with col2:
    st.metric("School Admins: Base", f"${int(round(sa['base_salary'].sum())):,}")
    st.metric("School Admins: Benefits", f"${int(round(sa['benefits'].sum())):,}")
    st.metric("School Admins: Total", f"${int(round(sa['total_compensation'].sum())):,}")

# === Teachers vs District Administrators Comparison (all three metrics) ===
st.subheader("Teachers vs District Administrators")
st.markdown("*Comparing teachers with superintendents, directors, and district-level staff*")

district_admins = filtered_data[filtered_data['functional_area'] == "District Administration"]
td = teachers
da = district_admins

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Teachers: Base", f"${int(round(td['base_salary'].sum())):,}")
    st.metric("Teachers: Benefits", f"${int(round(td['benefits'].sum())):,}")
    st.metric("Teachers: Total", f"${int(round(td['total_compensation'].sum())):,}")
with col2:
    st.metric("District Admins: Base", f"${int(round(da['base_salary'].sum())):,}" if len(da) > 0 else "N/A")
    st.metric("District Admins: Benefits", f"${int(round(da['benefits'].sum())):,}" if len(da) > 0 else "N/A")
    st.metric("District Admins: Total", f"${int(round(da['total_compensation'].sum())):,}" if len(da) > 0 else "N/A")

# Summary comparison table (all three metrics)
st.markdown("#### Summary: Teacher vs Administrator Compensation")

def safe_fmt(val):
    if pd.isna(val):
        return "N/A"
    return f"${int(round(val)):,}"

comparison_data = {
    'Role': ['Teachers', 'School Administrators', 'District Administrators'],
    'Employee Count': [len(teachers), len(school_admins), len(district_admins)],
    'Avg Base Salary': [safe_fmt(teachers['base_salary'].mean()), safe_fmt(school_admins['base_salary'].mean()), safe_fmt(district_admins['base_salary'].mean()) if len(district_admins) > 0 else "N/A"],
    'Avg Benefits': [safe_fmt(teachers['benefits'].mean()), safe_fmt(school_admins['benefits'].mean()), safe_fmt(district_admins['benefits'].mean()) if len(district_admins) > 0 else "N/A"],
    'Avg Total Compensation': [safe_fmt(teachers['total_compensation'].mean()), safe_fmt(school_admins['total_compensation'].mean()), safe_fmt(district_admins['total_compensation'].mean()) if len(district_admins) > 0 else "N/A"],
    'Total Base Salary': [safe_fmt(teachers['base_salary'].sum()), safe_fmt(school_admins['base_salary'].sum()), safe_fmt(district_admins['base_salary'].sum()) if len(district_admins) > 0 else "N/A"],
    'Total Benefits': [safe_fmt(teachers['benefits'].sum()), safe_fmt(school_admins['benefits'].sum()), safe_fmt(district_admins['benefits'].sum()) if len(district_admins) > 0 else "N/A"],
    'Total Compensation': [safe_fmt(teachers['total_compensation'].sum()), safe_fmt(school_admins['total_compensation'].sum()), safe_fmt(district_admins['total_compensation'].sum()) if len(district_admins) > 0 else "N/A"]
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# FTE vs Non-FTE Summary (all three metrics)
st.subheader("FTE vs Non-FTE Comparison")
fte_data = filtered_data[filtered_data['is_fte']]
non_fte_data = filtered_data[~filtered_data['is_fte']]
col1, col2 = st.columns(2)
with col1:
    st.metric("FTE Count", fte_data['employee_name'].nunique())
    st.metric("FTE Base Salary", f"${int(round(fte_data['base_salary'].sum())):,}")
    st.metric("FTE Benefits", f"${int(round(fte_data['benefits'].sum())):,}")
    st.metric("FTE Total Compensation", f"${int(round(fte_data['total_compensation'].sum())):,}")
with col2:
    st.metric("Non-FTE Count", non_fte_data['employee_name'].nunique())
    st.metric("Non-FTE Base Salary", f"${int(round(non_fte_data['base_salary'].sum())):,}")
    st.metric("Non-FTE Benefits", f"${int(round(non_fte_data['benefits'].sum())):,}")
    st.metric("Non-FTE Total Compensation", f"${int(round(non_fte_data['total_compensation'].sum())):,}")

# === Salary Distribution Analysis (all three metrics) ===
st.subheader("Compensation Distribution by Role")
st.markdown("This section shows how base salary, benefits, and total compensation are distributed within each functional area, helping identify compensation ranges and equity within roles.")

# Calculate stats by functional area for all three metrics
def comp_stats(df, col):
    return df.groupby('functional_area')[col].agg([
        'count', 'min', 'max', 'mean', 'median',
        lambda x: x.quantile(0.25),  # Q1
        lambda x: x.quantile(0.75),  # Q3
        'std'
    ]).round(0)

salary_stats = comp_stats(filtered_data, 'base_salary')
salary_stats.columns = ['Employee Count', 'Min Base', 'Max Base', 'Mean Base', 'Median Base', 'Q1 Base', 'Q3 Base', 'Std Base']
benefit_stats = comp_stats(filtered_data, 'benefits')
benefit_stats.columns = ['_b_count', 'Min Ben', 'Max Ben', 'Mean Ben', 'Median Ben', 'Q1 Ben', 'Q3 Ben', 'Std Ben']
total_stats = comp_stats(filtered_data, 'total_compensation')
total_stats.columns = ['_t_count', 'Min Total', 'Max Total', 'Mean Total', 'Median Total', 'Q1 Total', 'Q3 Total', 'Std Total']

stats = pd.concat([salary_stats, benefit_stats.iloc[:,1:], total_stats.iloc[:,1:]], axis=1)
stats['Base Range'] = stats['Max Base'] - stats['Min Base']
stats['Benefit Range'] = stats['Max Ben'] - stats['Min Ben']
stats['Total Range'] = stats['Max Total'] - stats['Min Total']
stats = stats.sort_values('Mean Total', ascending=False)

st.markdown("#### Compensation Statistics by Functional Area")
formatted_stats = stats.copy()
currency_cols = ['Min Base', 'Max Base', 'Mean Base', 'Median Base', 'Q1 Base', 'Q3 Base', 'Std Base',
                 'Min Ben', 'Max Ben', 'Mean Ben', 'Median Ben', 'Q1 Ben', 'Q3 Ben', 'Std Ben',
                 'Min Total', 'Max Total', 'Mean Total', 'Median Total', 'Q1 Total', 'Q3 Total', 'Std Total',
                 'Base Range', 'Benefit Range', 'Total Range']
for col in currency_cols:
    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"${int(x):,}")
st.dataframe(formatted_stats, use_container_width=True)

# Box Plot - Distribution by Role for all three metrics (with sampling for memory efficiency)
st.markdown("#### Compensation Distribution Box Plot")
st.markdown("Box plots show the spread of base salary, benefits, and total compensation within each role. The box shows the middle 50% of values, with the line inside showing the median.")

# Sample data for plotting if dataset is large
max_samples_per_group = 1000
sampled_data = filtered_data.copy()
if len(filtered_data) > 5000:
    sampled_data = filtered_data.groupby('functional_area').apply(
        lambda x: x.sample(min(len(x), max_samples_per_group), random_state=42)
    ).reset_index(drop=True)
    st.info(f"📊 Large dataset detected. Sampling {max_samples_per_group} employees per role for visualization.")

fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # Reduced figure size
for idx, (col, label, ax) in enumerate(zip(['base_salary', 'benefits', 'total_compensation'],
                                           ['Base Salary', 'Benefits', 'Total Compensation'], axs)):
    plot_data = []
    plot_labels = []
    for area in stats.index:
        role_data = sampled_data[sampled_data['functional_area'] == area][col]
        if len(role_data) > 0:
            plot_data.append(role_data)
            plot_labels.append(f"{area}\n(n={len(role_data)})")
    if len(plot_data) > 0:
        ax.boxplot(plot_data, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{label} Distribution', fontsize=10)
        ax.set_ylabel(f"{label} ($)", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.tight_layout()
st.pyplot(fig)

# Memory cleanup
plt.close(fig)
del sampled_data, plot_data, plot_labels, fig, axs

# Range Comparison Bar Charts (optimized for memory)
st.markdown("#### Compensation Range Analysis")
col1, col2, col3 = st.columns(3)

# Create range data once to avoid repeated processing
range_data = {
    'base': stats['Base Range'].apply(lambda x: float(str(x).replace("$","").replace(",",""))),
    'benefit': stats['Benefit Range'].apply(lambda x: float(str(x).replace("$","").replace(",",""))),
    'total': stats['Total Range'].apply(lambda x: float(str(x).replace("$","").replace(",","")))
}
y_pos = range(len(stats))

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
    ax.barh(y_pos, range_data['base'], color='skyblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats.index, fontsize=8)
    ax.set_xlabel('Base Salary Range ($)', fontsize=9)
    ax.set_title('Base Salary Range', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
    ax.barh(y_pos, range_data['benefit'], color='orange', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats.index, fontsize=8)
    ax.set_xlabel('Benefit Range ($)', fontsize=9)
    ax.set_title('Benefit Range', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
with col3:
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
    ax.barh(y_pos, range_data['total'], color='green', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats.index, fontsize=8)
    ax.set_xlabel('Total Compensation Range ($)', fontsize=9)
    ax.set_title('Total Compensation Range', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Memory cleanup
del range_data, y_pos


# Sample job titles for each role
st.subheader("Sample Job Titles by Functional Area")
sample_titles = df.groupby('functional_area')['title'].unique().apply(lambda x: ', '.join(list(pd.Series(x).dropna().unique())[:5]))
st.dataframe(sample_titles.reset_index().rename(columns={'title': 'Sample Titles'}))

# --- Enhanced Top Earners Section (dual-metric display) ---
st.subheader("Top Earners in District")

# --- Enhanced Top Earners Controls ---
sort_by = st.selectbox(
    "Sort by",
    ["Total Compensation", "Base Salary", "Benefits"],
    index=0,
    help="Select which column to rank top earners by"
)
sort_order = st.selectbox(
    "Order", ["Highest to Lowest", "Lowest to Highest"], index=0
)
top_n = st.selectbox(
    "Show Top", [10, 20, 50, 100], index=1
)
sort_col_map = {
    "Total Compensation": "total_compensation",
    "Base Salary": "base_salary",
    "Benefits": "benefits"
}
sort_col = sort_col_map[sort_by]
ascending = (sort_order == "Lowest to Highest")

top_earners = employee_breakdown.sort_values(by=sort_col, ascending=ascending).head(top_n).copy()

top_earners_display = top_earners[[
    'employee_initials', 'functional_area', 'employment_status',
    'base_salary', 'benefits', 'total_compensation', 'benefits_percentage'
]].rename(columns={
    'employee_initials': 'Employee',
    'functional_area': 'Role',
    'employment_status': 'Employment Status',
    'base_salary': 'Base Salary',
    'benefits': 'Benefits',
    'total_compensation': 'Total Compensation',
    'benefits_percentage': 'Benefits %'
})

for col in ['Base Salary', 'Benefits', 'Total Compensation']:
    top_earners_display[col] = top_earners_display[col].apply(lambda x: f"${int(round(x)):,}")
top_earners_display['Benefits %'] = top_earners_display['Benefits %'].apply(lambda x: f"{x:.1f}%")

st.dataframe(top_earners_display, use_container_width=True)
# --- End Top Earners Section ---

# Raw data download with privacy protection (all three metrics)
st.subheader("Download Cleaned Data")
st.markdown("*Download includes privacy protection - employee names converted to initials*")

# Create privacy-protected download version
download_data = filtered_data[['employee_initials', 'functional_area', 'is_fte', 'base_salary', 'benefits', 'total_compensation']].copy()
download_data = download_data.rename(columns={'employee_initials': 'employee'})

st.download_button(
    "Download as CSV",
    download_data.to_csv(index=False),
    f"alpine_district_salary_analysis_{selected_year}.csv",
    "text/csv",
    help="CSV file with employee names converted to initials for privacy"
)

st.subheader("Crosscheck with Financial Report")
st.markdown("""
- **Report FTE**: 8,231  
- **Your dataset names**: 14,897 (includes duplicates, subs, part-time)  
- **Report: 61.9% to Instruction**  
- **App: {:.1f}% to Teachers (Total Compensation)**  
- **App: {:.1f}% to Teachers (Base Salary)**  
""".format(
    (teachers['total_compensation'].sum() / filtered_data['total_compensation'].sum()) * 100 if filtered_data['total_compensation'].sum() > 0 else 0,
    (teachers['base_salary'].sum() / filtered_data['base_salary'].sum()) * 100 if filtered_data['base_salary'].sum() > 0 else 0
))
