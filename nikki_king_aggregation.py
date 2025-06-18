import pandas as pd

# NIKKI KING's raw records
nikki_records = [
    {"employee_name": "KING, NIKKI", "title": "Hourly Cert. Salary", "net_amount": 524.37, "org2": "099 - $4200 Teacher Salary Adj"},
    {"employee_name": "KING, NIKKI", "title": "Certified Teacher Salary", "net_amount": 48178.81, "org2": "125 - Freedom Elementary"},
    {"employee_name": "KING, NIKKI", "title": "Certified Teacher Salary", "net_amount": 69122.84, "org2": "125 - Freedom Elementary"},
    {"employee_name": "KING, NIKKI", "title": "Stipends/Honorariums", "net_amount": 504.43, "org2": "080 - All Schools"},
    {"employee_name": "KING, NIKKI", "title": "Stipends/Honorariums", "net_amount": 1600.00, "org2": "080 - All Schools"},
    {"employee_name": "KING, NIKKI", "title": "Hourly Cert. Salary", "net_amount": 165.71, "org2": "099 - $4200 Teacher Salary Adj"},
    {"employee_name": "KING, NIKKI", "title": "Extracurricular Addenda", "net_amount": 500.00, "org2": "125 - Freedom Elementary"}
]

df = pd.DataFrame(nikki_records)

# Classification function (same as in app.py)
def classify(row):
    title = row['title'].lower()
    if "teacher" in title or "cert" in title:
        return "Instruction"
    elif "principal" in title or "school admin" in title:
        return "School Administration"
    elif any(x in title for x in ["superintendent", "director", "district admin"]):
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

# FTE determination function (same as in app.py)
def is_fte(row):
    title = row['title'].lower()
    if any(x in title for x in ["hourly", "substitute", "part-time", "temp"]):
        return False
    return True

# Apply classification to first record (primary position)
df['functional_area'] = df.apply(classify, axis=1)
df['is_fte'] = df.apply(is_fte, axis=1)

# Aggregate by employee
aggregated = df.groupby('employee_name').agg({
    'net_amount': 'sum',
    'functional_area': 'first',  # Take the first classification (primary role)
    'is_fte': lambda x: any(x)   # True if any position is FTE
}).reset_index()

# Add row number for target schema format
aggregated.insert(0, '', range(len(aggregated)))

# Format net_amount as currency string
aggregated['net_amount'] = aggregated['net_amount'].apply(lambda x: f"${x:,.0f}")

print("NIKKI KING aggregated record:")
print(aggregated.to_csv(index=False, quoting=1))

print("\nFormatted for your target schema:")
for _, row in aggregated.iterrows():
    print(f'{row.iloc[0]},"{row["employee_name"]}",{row["functional_area"]},{str(row["is_fte"]).lower()},"{row["net_amount"]}"')
