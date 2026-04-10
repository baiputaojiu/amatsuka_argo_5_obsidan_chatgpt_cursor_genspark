from datetime import datetime, time

def day_range(start_date,end_date):
    return datetime.combine(start_date,time.min), datetime.combine(end_date,time.max)
