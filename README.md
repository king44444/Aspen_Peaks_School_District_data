# Alpine School District - Salary Analysis Dashboard

*Created by Michael King*

A comprehensive Streamlit web application for analyzing employee salary data from Alpine School District. This tool provides detailed insights into salary distribution, equity analysis, and compensation patterns across different roles and functional areas.

**🚨 2024 Update:** This analysis is particularly critical as it represents the final year before Alpine School District undergoes significant reorganization and potential district splitting. The 2024 data provides crucial baseline insights for community stakeholders.

## 📊 Data Source & Methodology

### Data Source
All salary data in this analysis comes directly from **Utah's official transparency portal** at [transparent.utah.gov](https://transparent.utah.gov). This ensures the underlying financial information is accurate and represents official state records of public employee compensation.

### Important Disclaimer
While the raw salary data is official state information, the **classification and grouping of employees into functional areas** (such as "Instruction," "Administration," etc.) represents the author's interpretation based on job titles and descriptions. These categorizations, while systematic and logical, may not perfectly align with the district's internal organizational structure. Users should consider these groupings as analytical tools rather than official district classifications.

### Geographic Context
This analysis focuses on **Alpine School District** salary data, with an optional filter for **Aspen Peaks District** schools. The Aspen Peaks filter represents a geographic subset of schools within the Alpine School District boundaries, specifically those serving the communities around American Fork, Lehi, Highland, and surrounding areas.

## 🚀 Quick Start

### For Non-Technical Users
1. **View Results**: If someone has already set up the application, simply visit the web address they provide
2. **Interact**: Use the sidebar filters to focus on specific employee types or geographic areas
3. **Explore**: Scroll through different sections to see salary comparisons, distributions, and top earners
4. **Download**: Export the data as CSV for your own analysis

### For Technical Users
```bash
# Clone and setup
git clone [your-repo-url]
cd alpine-school-district-analysis
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🚀 Features

### 🎯 New: Community-Focused Analysis
- **Key Takeaways Section**: Executive summary highlighting critical insights for community members
- **Teacher Pay Equity Focus**: Clear analysis of teacher compensation compared to administrators
- **Budget Priority Analysis**: Shows percentage of district budget allocated to instruction vs. administration
- **Community Context**: Interprets data implications for teacher retention, recruitment, and district priorities

### 🔒 Privacy Protection
- **Employee Name Protection**: All employee names converted to initials (e.g., "King, Nikki" → "K, N")
- **Privacy-Protected Downloads**: CSV exports include only initials, not full names
- **Secure Data Display**: Top earners and all employee tables show initials only

### 📅 Multi-Year Analysis Capability
- **Historical Data Support**: Load and analyze data from 2022, 2023, and 2024
- **Year Selection**: Choose specific years for focused analysis
- **2024 Priority**: Defaults to 2024 data as the critical baseline before district reorganization

### Core Analysis
- **Employee Classification**: Automatic categorization into functional areas (Instruction, Administration, Support, etc.)
- **FTE vs Non-FTE Analysis**: Comparison between full-time and part-time/substitute employees
- **Salary Distribution**: Detailed statistical analysis of salary ranges within each role
- **Management vs Frontline Comparison**: Direct comparison between teachers and administrators

### Advanced Visualizations
- **Box Plots**: Show salary distribution spread within each functional area
- **Salary Range Analysis**: Horizontal bar charts showing salary ranges by role
- **Distribution Histograms**: Frequency distribution of salaries within each role
- **Scatter Plots**: Relationship between employee count and average salary
- **Equity Insights**: Identification of roles with highest variability and most equitable pay

### Interactive Features
- **Aspen Peaks District Filter**: Focus analysis on specific school boundaries
- **Role-based Filtering**: Select specific employee types for targeted analysis
- **Dynamic Metrics**: Real-time updates based on applied filters
- **Enhanced Data Export**: Privacy-protected CSV downloads with year-specific naming

## 📊 Sample Visualizations

The dashboard includes:
- Total salary expenditure by employee type
- Average salary comparisons across roles
- Salary equity analysis with coefficient of variation
- Top earners identification
- Statistical breakdowns (min, max, median, quartiles)

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages
```bash
pip install streamlit pandas matplotlib seaborn
```

### Setup
1. Clone this repository:
```bash
git clone [your-repo-url]
cd alpine-school-district-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your data file is in the correct location:
```
data/Alpine_School_District_2024.csv
```

## 🚀 Usage

### Launch the Application
```bash
streamlit run app.py
```

The application will automatically open in your web browser at `http://localhost:8501`

### Navigation
1. **Sidebar Controls**: Use filters to focus on specific employee types or districts
2. **Main Dashboard**: Scroll through different analysis sections
3. **Interactive Charts**: Hover over visualizations for detailed information
4. **Data Export**: Use the download button to export filtered data

### Key Sections
- **Overview Metrics**: High-level employee and salary statistics
- **Role Comparisons**: Analysis across different functional areas
- **Management vs Frontline**: Direct comparison of administrative and teaching staff
- **Salary Distribution**: Comprehensive statistical analysis of pay equity
- **Top Earners**: Identification of highest-paid employees

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── data/
│   ├── Alpine_School_District_2024.csv  # Primary dataset
│   └── sample_data.txt             # Sample data for testing
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## 📈 Data Structure

The application expects a CSV file with the following key columns:
- `employee_name`: Employee identifier
- `title`: Job title/position
- `net_amount`: Salary amount
- `org2`: School/organization assignment
- `description`: Additional role description
- `fiscal_year`: Year of data

## 🔍 Analysis Methodology

### Employee Classification Logic
Employees are automatically classified into functional areas based on keywords in job titles and descriptions. This classification system was developed by analyzing common patterns in the data:

**Instruction**
- Keywords: "teacher", "cert" (certified)
- Examples: "Elementary Teacher", "Secondary Teacher", "Certified Instructor"

**School Administration**
- Keywords: "principal", "school admin"
- Examples: "Elementary Principal", "Assistant Principal"

**District Administration**
- Keywords: "superintendent", "director", "district admin", "board"
- Examples: "Superintendent", "District Director", "Board Member"

**Student Support**
- Keywords: "counselor", "psychologist", "speech"
- Examples: "School Counselor", "Speech Therapist", "School Psychologist"

**Operations & Maintenance**
- Keywords: "custodian", "maintenance", "facilities"
- Examples: "Head Custodian", "Maintenance Worker", "Facilities Manager"

**Transportation**
- Keywords: "transport"
- Examples: "Bus Driver", "Transportation Supervisor"

**Food Services**
- Keywords: "food", "nutrition"
- Examples: "Kitchen Manager", "Nutrition Assistant"

**Other**
- Any employee not fitting the above categories

### FTE Classification
Full-time equivalent (FTE) status determined by job title keywords:
- **Non-FTE**: Contains "hourly", "substitute", "part-time", or "temp"
- **FTE**: All other employees

### Statistical Measures
- **Standard Deviation**: Measure of salary variability within roles
- **Coefficient of Variation**: Normalized measure of salary equity (lower = more equitable)
- **Quartiles**: 25th, 50th (median), and 75th percentile salary ranges
- **Salary Range**: Difference between minimum and maximum salaries

### Data Limitations & Interpretation Guidelines

**Strengths of This Analysis:**
- Based on official state salary data
- Comprehensive coverage of all district employees
- Systematic classification methodology
- Multiple analytical perspectives (role, geography, FTE status)

**Limitations to Consider:**
- Employee classifications are interpretive, not official district categories
- Job titles may not fully reflect actual responsibilities
- Part-time vs. full-time distinctions based on title keywords only
- No adjustment for experience, education, or certification levels
- Geographic school assignments may not perfectly match community boundaries

**Best Practices for Interpretation:**
- Use salary ranges and medians rather than focusing on individual outliers
- Compare similar-sized employee groups for meaningful insights
- Consider the context that education compensation includes benefits not captured here
- Cross-reference findings with official district budget documents when possible

## 🎯 Use Cases

### For School Board Members
- Budget allocation analysis across functional areas
- Salary equity assessment within roles
- Comparison framework for policy decisions
- Transparency tool for public accountability

### For Human Resources
- Pay scale analysis and benchmarking
- Identification of potential equity issues requiring review
- Workforce composition insights
- Data-driven compensation planning

### For Public Transparency
- Clear visualization of public salary expenditures
- Understanding of district priorities through compensation patterns
- Accessible format for community engagement

### For Researchers & Policy Analysts
- Educational finance analysis framework
- Comparative studies across districts (when applied to multiple datasets)
- Policy impact assessment tool
- Template for replication in other districts

## 🔧 Customization

### Adding New Functional Areas
Modify the `classify()` function in `app.py` to add new employee categories:

```python
def classify(row):
    title = row['title'].lower()
    if "new_role" in title:
        return "New Category"
    # ... existing logic
```

### Adjusting School Lists
Update the `aspen_peaks_schools` list to modify geographic filtering:

```python
aspen_peaks_schools = [
    "Your School Name",
    # ... other schools
]
```

### Modifying Visualizations
All charts can be customized by modifying the matplotlib/seaborn parameters in the respective sections.

## 📊 Understanding the Results

### Most Reliable Insights
- **Total compensation by functional area**: High confidence due to direct data aggregation
- **Employee count distributions**: Accurate representation of workforce composition
- **Salary ranges within roles**: Good indicator of pay equity and structure

### Insights Requiring Caution
- **Individual employee classifications**: May not match official district organization
- **Comparisons between very different roles**: Context matters (education, experience, responsibilities)
- **Geographic analysis**: School assignments may not perfectly reflect community served

### Contextual Considerations
- **Total Compensation**: The figures shown include both salaries and employee benefits (approximately 70% salary, 30% benefits based on sample data analysis)
- Teacher salaries typically follow structured pay scales based on education and experience
- Administrative roles may have different compensation structures and expectations
- Seasonal and substitute workers create natural salary distribution patterns

## 📊 Sample Data

The repository includes sample data (`data/sample_data.txt`) for testing purposes. This contains a subset of the full dataset structure to help understand the expected format without using actual employee information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

See `requirements.txt` for a complete list of Python dependencies:
- streamlit
- pandas
- matplotlib
- seaborn

## 📝 License & Usage

This project is intended for educational and transparency purposes. The analysis methodology and code are freely available for adaptation to other school districts or public sector organizations.

**Data Privacy Compliance**: When using this tool with actual salary data, ensure compliance with local data privacy laws, public records regulations, and ethical guidelines for handling employee information.

## 🆘 Troubleshooting

### Common Issues

**"streamlit command not found"**
```bash
pip install streamlit
```

**"No module named 'seaborn'"**
```bash
pip install seaborn matplotlib pandas
```

**Data loading errors**
- Ensure CSV file is in `data/` directory
- Check that column names match expected format (`employee_name`, `title`, `net_amount`, `org2`, `description`)
- Verify file encoding (should be UTF-8)

**Performance issues with large datasets**
- Large datasets may require increased memory allocation
- Consider filtering data before analysis for files with >50,000 records
- Use the geographic filter to focus on smaller subsets for initial analysis

**Classification seems incorrect**
- Review the `classify()` function logic in `app.py`
- Consider that job titles may not perfectly reflect roles
- Classifications are interpretive tools, not definitive categorizations

### Getting Help
- Check the Streamlit documentation: https://docs.streamlit.io/
- Review matplotlib documentation for chart customization: https://matplotlib.org/
- Open an issue in this repository for specific problems
- For data interpretation questions, consider consulting with district HR or finance departments

## 🎉 Acknowledgments

- **Data Source**: Utah State Government via [transparent.utah.gov](https://transparent.utah.gov)
- **Analysis & Development**: Michael King
- **Built with**: [Streamlit](https://streamlit.io/) for the web interface
- **Visualizations**: [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)

---

**Final Note**: This analysis tool demonstrates the power of public data transparency. While every effort has been made to ensure accurate analysis, users should remember that data interpretation involves subjective elements. The goal is to provide useful insights while maintaining appropriate context about the limitations and methodology involved.

For questions about the data source or official district information, please contact Alpine School District directly. For questions about this analysis tool or methodology, feel free to open an issue in this repository.
