from datetime import datetime
from dateutil.relativedelta import relativedelta

def parse_date(date_str):
    # Define month names and their corresponding numbers
    month_names = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
    
    # Replace month names with numbers
    for month_name, month_number in month_names.items():
        date_str = date_str.lower().replace(month_name, month_number)
    
    # Define possible date formats
    formats = ['%Y%m%d', '%m%d%Y', '%d%m%Y', '%m%d']
    
    # Try to parse the date with each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    
    # If no format matches, return None
    return None

def calculate_date_difference(date_str):
    # Parse the date
    date = parse_date(date_str)
    
    # If the date could not be parsed, return an error message
    if date is None:
        return 'Invalid date format'
    
    # Calculate the difference between the parsed date and the current date
    difference = date - datetime.now()
    
    # If the difference is negative, return 'expired'
    if difference.days < 0:
        return 'expired'
    
    # Otherwise, return the difference in days
    return date.strftime('%Y-%m-%d')

# Test the function
print(calculate_date_difference('0722'))  # Should print the number of days between 2023-05-22 and the current date, or 'expired' if the current date is later
# print(calculate_date_difference('May2223'))  # Should print the number of days between 2023-05-22 and the current date, or 'expired' if the current date is later
# print(calculate_date_difference('220523'))  # Should print the number of days between 2023-05-22 and the current date, or 'expired' if the current date is later
# print(calculate_date_difference('0522'))  # Should print the number of days between this year's 05-22 and the current date, or 'expired' if the current date is later
