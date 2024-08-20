import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # Handle missing values
    df.dropna(inplace=True)
    
    # Convert categorical columns to appropriate types
    bool_columns = ['Scholarship', 'Hipertension', 'Alcoholism', 'Diabetes', 'SMS_received', 'No-show']
    for column in bool_columns:
        df[column] = df[column].astype('bool')
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle negative ages
    df.loc[df['Age'] <= 0, 'Age'] = df['Age'].mean()
    
    # Remove invalid data
    df = df[df['ScheduledDay'] <= df['AppointmentDay']]
    
    return df

def feature_engineering(df):
    # Convert dates to datetime
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    # Create new features
    df['ScheduledHour'] = df['ScheduledDay'].dt.hour
    df['ScheduledMonth'] = df['ScheduledDay'].dt.month
    df['ScheduledDayOfWeek'] = df['ScheduledDay'].dt.dayofweek
    df['AwaitingTimeDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    
    # Convert gender to binary
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'F' else 0)
    
    # Drop unnecessary columns
    df.drop(['ScheduledDay', 'AppointmentDay', 'PatientId', 'AppointmentID'], axis=1, inplace=True)
    
    return df

def preprocess_data(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    return df
