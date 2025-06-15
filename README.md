# Alpine School District - Salary Analysis Dashboard

A comprehensive Streamlit web application for analyzing employee salary data from Alpine School District. This tool provides detailed insights into salary distribution, equity analysis, and compensation patterns across different roles and functional areas.

## 🚀 Features

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
- **Data Export**: Download cleaned data as CSV for further analysis

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

### Employee Classification
Employees are automatically classified into functional areas based on job titles:
- **Instruction**: Teachers and certified instructional staff
- **School Administration**: Principals and school-level administrators
- **District Administration**: Superintendents, directors, district-level staff
- **Student Support**: Counselors, psychologists, speech therapists
- **Operations & Maintenance**: Custodial and facilities staff
- **Transportation**: Bus drivers and transportation staff
- **Food Services**: Nutrition and cafeteria staff
- **Other**: Employees not fitting other categories

### FTE Classification
Full-time equivalent (FTE) status determined by job title keywords:
- **Non-FTE**: Contains "hourly", "substitute", "part-time", or "temp"
- **FTE**: All other employees

### Statistical Measures
- **Standard Deviation**: Measure of salary variability within roles
- **Coefficient of Variation**: Normalized measure of salary equity
- **Quartiles**: 25th, 50th (median), and 75th percentile salary ranges
- **Salary Range**: Difference between minimum and maximum salaries

## 🎯 Use Cases

### For School Board Members
- Budget allocation analysis across functional areas
- Salary equity assessment
- Comparison with state/national benchmarks

### For Human Resources
- Pay scale analysis and recommendations
- Identification of potential equity issues
- Workforce composition insights

### For Public Transparency
- Clear visualization of public salary expenditures
- Understanding of district priorities through compensation analysis

### For Researchers
- Educational finance analysis
- Comparative studies across districts
- Policy impact assessment

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

## 📊 Sample Data

The repository includes sample data (`data/sample_data.txt`) for testing purposes. This contains a subset of the full dataset structure to help understand the expected format.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Requirements

See `requirements.txt` for a complete list of Python dependencies.

## 📝 License

This project is intended for educational and transparency purposes. Please ensure compliance with local data privacy and public records laws when using with actual salary data.

## 🆘 Troubleshooting

### Common Issues

**"streamlit command not found"**
```bash
pip install streamlit
```

**"No module named 'seaborn'"**
```bash
pip install seaborn
```

**Data loading errors**
- Ensure CSV file is in `data/` directory
- Check that column names match expected format
- Verify file encoding (should be UTF-8)

**Performance issues**
- Large datasets may require increased memory
- Consider filtering data before analysis for very large files

### Getting Help
- Check the Streamlit documentation: https://docs.streamlit.io/
- Review matplotlib documentation for chart customization
- Open an issue in this repository for specific problems

## 🎉 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Visualizations powered by [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- Data analysis using [Pandas](https://pandas.pydata.org/)

---

**Note**: This tool is designed for transparency and analysis purposes. Always ensure compliance with local privacy laws and regulations when working with employee salary data.
