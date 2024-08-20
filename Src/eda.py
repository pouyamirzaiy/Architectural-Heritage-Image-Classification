import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(df):
    df['No-show'].value_counts().plot(kind='bar')
    plt.title('Patients Show-Up for Their Appointment')
    plt.xlabel('Show-Up')
    plt.ylabel('Counts')
    plt.legend(['Show-Up', 'No Show-Up'], loc='upper right', title='Key')
    plt.show()

def plot_sms_received(df):
    df['SMS_received'].value_counts().plot(kind='bar')
    plt.title('SMS Received by Patients')
    plt.xlabel('SMS Received')
    plt.ylabel('Counts')
    plt.legend(['No SMS Received', 'SMS Received'], loc='upper right', title='Key')
    plt.show()

def plot_awaiting_time_distribution(df):
    sns.histplot(df['AwaitingTimeDays'], bins=50)
    plt.title('Distribution of Awaiting Time')
    plt.xlabel('Awaiting Time Days')
    plt.show()

def plot_age_distribution(df):
    sns.histplot(df['Age'], bins=50, kde=True)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.show()

def eda(df):
    plot_class_distribution(df)
    plot_sms_received(df)
    plot_awaiting_time_distribution(df)
    plot_age_distribution(df)
