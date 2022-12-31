import pandas as pd
import numpy as np
import Classification as cs
from pandas_ods_reader import read_ods 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests, zipfile, io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,classification_report, confusion_matrix, accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd #Importing Pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier,plot_importance
import RandomForestRegression as rfr
import lasso as ls
#import easygui_house as easy
import tkinterpy as tkpy
#from tkinterpy import df
from scipy.stats import skew
import easygui as eg
def main():
    

    ### Housing Data ###
    housing_19 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2019.csv',nrows=70000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_19.columns = column_names

    housing_20 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv',nrows=70000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_20.columns = column_names

    housing_21 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2021.csv',nrows=70000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_21.columns = column_names

    #housing_22 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2022.csv.csv',usecols =[0,1,2,3,4,5,6,12,13],header = None))
    #column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    #housing_22.columns = column_names

    df1=pd.concat([housing_19,housing_20,housing_21])


    ### Housing Data ###

    ### Housing Data ###
    housing_19 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2019.csv',nrows=40000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_19.columns = column_names

    housing_20 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv',nrows=40000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_20.columns = column_names

    housing_21 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2021.csv',nrows=40000,usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    housing_21.columns = column_names

    #housing_22 = pd.DataFrame(pd.read_csv('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2022.csv.csv',usecols =[0,1,2,3,4,5,6,12,13],header = None))
    #column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    #housing_22.columns = column_names

    df1=pd.concat([housing_19,housing_20,housing_21])

    #df1 = pd.DataFrame(pd.read_excel('pp-2022.xlsx',usecols =[0,1,2,3,4,5,6,12,13],header = None))
    column_names=['sale_id','price','date','postcode','house_type','new_build','lease_type','Borough_name','County']
    df1.columns = column_names
    df1['date'] = pd.to_datetime(df1['date'])
    df1 = df1.loc[(df1['date'] >= '2019-01-01') & (df1['date'] < '2022-08-01')]
    df1['year'] = pd.DatetimeIndex(df1['date']).year
    df1['month'] = pd.DatetimeIndex(df1['date']).month
    df1 = df1.applymap(lambda s: s.lower() if type(s) == str else s)
    #df1['[price]'] = df1['price'].astype('int')
    c=['bedford','buckinghamshire','cambridgeshire','cheshire east', 'cheshire west and chester','cleveland','cornwall','cumbria','derbyshire','devon','dorset','durham','east sussex','essex','gloucestershire','greater london','greater manchester','hampshire','hertfordshire','kent','lancashire','leicestershire','lincolnshire', 'north east lincolnshire', 'north lincolnshire', 'merseyside','norfolk','north yorkshire','northamptonshire', 'north northamptonshire','northumberland','nottinghamshire','oxfordshire','shropshire','somerset', 'north somerset','south yorkshire','south gloucestershire', 'staffordshire','suffolk','surrey','tyne and wear','warwickshire','west berkshire','west midlands','west sussex','west yorkshire','wiltshire','worcestershire']
    df1=df1[df1['County'].isin(c)]
    df1=df1[df1['Borough_name'].isin(c_b)]

    df1 = df1[df1['house_type'] != 'o'].copy()
    #df1['price'] = df['Weight'].astype(int)


    ### Clinics Data ###
    df2 = pd.DataFrame(pd.read_csv ('Clinics.csv',encoding= 'latin1',sep='¬'))
    df2.rename(columns = {'City':'Borough_name'}, inplace = True)
    df2=df2.drop_duplicates(subset='Address1',keep="last")
    #df2 = df2.dropna(subset=['Borough_name'])
    df2 = df2.applymap(lambda s: s.lower() if type(s) == str else s)
    df2=df2[df2['Borough_name'].isin(c_b)]
    #clinics_count= pd.value_counts(df2['Borough_name'].values).rename_axis('Borough_name').reset_index(name='clinics_count')
    df2 = df2.groupby(['Borough_name'])['Borough_name'].count().reset_index(name='clinics_count')
    #df2 = df.groupby(['Courses','Duration']).size().reset_index(name='counts')
    df2=pd.DataFrame(df2)


    ### School Data ###
    school_url='https://www.find-school-performance-data.service.gov.uk/download-data?download=true&regions=0&filters=GIAS&fileformat=xls&year=2021-2022&meta=false'
    df3= pd.DataFrame(pd.read_excel(school_url))
    df3.rename(columns = {'TOWN':'Borough_name'}, inplace = True)
    df3=df3.drop_duplicates(subset='LOCALITY',keep="last")
    df3 = df3.applymap(lambda s: s.lower() if type(s) == str else s)
    df3=df3[df3['Borough_name'].isin(c_b)]
    df3 = df3.groupby(['Borough_name'])['Borough_name'].count().reset_index(name='schools_count')


    ### Hospital Data ###
    df4 = pd.DataFrame(pd.read_csv ('Hospital.csv',encoding= 'latin1',sep='¬'))
    df4.rename(columns = {'City':'Borough_name'}, inplace = True)
    df4=df4.drop_duplicates(subset='Address1',keep="last")
    df4 = df4.applymap(lambda s: s.lower() if type(s) == str else s)

    df4=df4[df4['Borough_name'].isin(c_b)]
    #clinics_count= pd.value_counts(df2['Borough_name'].values).rename_axis('Borough_name').reset_index(name='clinics_count')
    df4 = df4.groupby(['Borough_name'])['Borough_name'].count().reset_index(name='hospital_count')


    ### Pubs Data ###

    pubs_url = 'https://www.getthedata.com/downloads/open_pubs.csv.zip'
    filename = 'open_pubs.csv'

    r = requests.get(pubs_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

    columns=['ID','pub_name','area_name','postcode','a','x','y','z','Borough_name']
    df5 = pd.read_csv(filename, sep=',',header=None,names=columns)
    df5=df5.drop_duplicates(subset='area_name',keep="last")
    df5 = df5.applymap(lambda s: s.lower() if type(s) == str else s)
    df5=df5[df5['Borough_name'].isin(c_b)]
    #clinics_count= pd.value_counts(df2['Borough_name'].values).rename_axis('Borough_name').reset_index(name='clinics_count')
    df5 = df5.groupby(['Borough_name'])['Borough_name'].count().reset_index(name='pubs_count')



    ### GP Data ###
    df6 = pd.DataFrame(pd.read_csv ('GPPractices.csv',encoding= 'latin1',sep='¬'))
    df6.rename(columns = {'City':'Borough_name'}, inplace = True)
    df6 = df6.applymap(lambda s: s.lower() if type(s) == str else s)
    df6=df6[df6['Borough_name'].isin(c_b)]
    #clinics_count= pd.value_counts(df2['Borough_name'].values).rename_axis('Borough_name').reset_index(name='clinics_count')
    df6 = df6.groupby(['Borough_name'])['Borough_name'].count().reset_index(name='gp_count')



    ### Deprivation Data ###

    deprivation_url='https://opendata.camden.gov.uk/api/views/8x5x-eu22/rows.csv'
    df7 = pd.DataFrame(pd.read_csv(deprivation_url, usecols =[3,10,14,26]))
    df7.rename(columns = {'Local Authority District Name':'Borough_name'}, inplace=True)
    df7 = df7.applymap(lambda s: s.lower() if type(s) == str else s)
    df7=df7[df7['Borough_name'].isin(c_b)]
    ### Population Data ###

    df8=pd.DataFrame(pd.read_excel('census2021firstresultsenglandwales1.xlsx',sheet_name='P01',skiprows=6,usecols=[1,2]))
    df8.rename(columns = {'Area name':'Borough_name','All persons':'population'}, inplace = True)### Rename primary key ###
    df8 = df8.applymap(lambda s: s.lower() if type(s) == str else s)
    df8=df8[df8['Borough_name'].isin(c_b)]
    ### Parks Data ###

    df9=pd.DataFrame(pd.read_excel('ospublicgreenspacereferencetables.xlsx',sheet_name='LAD Parks only',usecols=[5,6,8]))
    df9.rename(columns = {'LAD name':'Borough_name','Average distance to nearest Park, Public Garden, or Playing Field (m)':'Avg_distance_to_park','Average number of  Parks, Public Gardens, or Playing Fields within 1,000 m radius':'avg_number_of_parks'}, inplace = True)### Rename primary key ###
    df9 = df9.applymap(lambda s: s.lower() if type(s) == str else s)
    df9=df9[df9['Borough_name'].isin(c_b)]

    df11=pd.DataFrame(pd.read_excel('https://simplemaps.com/static/data/country-cities/gb/gb.xlsx',usecols=[0,1,2]))
    df11.rename(columns={'city':'Borough_name'},inplace=True)
    df11 = df11.applymap(lambda s: s.lower() if type(s) == str else s)
    df11=df11.query("Borough_name in @c_b")
    df11=df11[df11['Borough_name'].isin(c_b)]
    #clinics_count= pd.value_counts(df2['Borough_name'].values).rename_axis('Borough_name').reset_index(name='clinics_count')

    df12 = pd.DataFrame(pd.read_csv('Sales-2022-09.csv',usecols =[0,1,3],header = 0))
    df12.rename(columns = {'Region_Name':'Borough_name','Date':'date','Sales_volume':'sales_volume'}, inplace = True)

    df12['date'] = pd.to_datetime(df12['date'])

    df12 = df12.loc[(df12['date'] >= '2019-01-01') & (df12['date'] < '2022-08-01')]
    df12['year'] = pd.DatetimeIndex(df12['date']).year
    df12['month'] = pd.DatetimeIndex(df12['date']).month
    df12 = df12.applymap(lambda s: s.lower() if type(s) == str else s)
    #df1['[price]'] = df1['price'].astype('int')

    df12=df12[df12['Borough_name'].isin(c_b)]

    df13 = pd.DataFrame(pd.read_csv('Cash-mortgage-sales-2022-09.csv',usecols =[0,1,7,12],header = 0))
    df13.rename(columns = {'Region_Name':'Borough_name','Date':'date','Sales_volume':'sales_volume'}, inplace = True)

    df13['date'] = pd.to_datetime(df13['date'])

    df13 = df13.loc[(df13['date'] >= '2019-01-01') & (df13['date'] < '2022-08-01')]
    df13['year'] = pd.DatetimeIndex(df13['date']).year
    df13['month'] = pd.DatetimeIndex(df13['date']).month
    df13 = df13.applymap(lambda s: s.lower() if type(s) == str else s)
    #df1['[price]'] = df1['price'].astype('int')

    df13=df13[df13['Borough_name'].isin(c_b)]


    final_data=pd.merge(df1, df12, how="left", on=['month','year','Borough_name'])
    final_data=pd.merge(final_data, df13, how="left", on=['month','year','Borough_name'])
    final_data = pd.merge(final_data, df2, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df3, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df4, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df5, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df6, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df7, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df8, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df9, how="left", on="Borough_name")
    final_data=pd.merge(final_data, df11, how="left", on="Borough_name")


    #df=pd.DataFrame(pd.read_excel('housing_test.xlsx', index_col=0))
    df=final_data.copy()
    df=df.drop_duplicates(subset='sale_id',keep="last")
    df=df.drop_duplicates()
    #df['price'] = df['price'].astype('int')

    #df=pd.DataFrame(pd.read_excel('housing_test.xlsx', index_col=0))
    #df=df.dropna()
    df.fillna(df.median(), inplace=True)

    cols = ['price',  'Cash_Sales_Volume', 'Mortgage_Sales_Volume',
            'Income Decile', 'Emplyment Decile', 'Crime Decile',
           'population', 'Avg_distance_to_park', 'avg_number_of_parks']# one or more

    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    df=df[['Borough_name','County','price', 'house_type', 'new_build',
           'lease_type', 'Cash_Sales_Volume', 'Mortgage_Sales_Volume',
           'clinics_count', 'schools_count', 'hospital_count', 'pubs_count',
           'gp_count', 'Income Decile', 'Emplyment Decile', 'Crime Decile',
           'population', 'Avg_distance_to_park', 'avg_number_of_parks', 'lat',
           'lng','year','month']]

    #le = LabelEncoder()
    #for column in df[['lease_type','house_type','new_build']].columns:
    #    df[column]=le.fit_transform(df[column].values)
    for column in df[['lease_type','house_type','new_build']].columns:
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes


    df.drop(df[df['price'] <= 10000].index, inplace = True)
    df.drop(df[df['price'] >= 700000].index, inplace = True) 


    df.dropna()
    df.to_csv('cleaned_data1.csv')
    df=df.copy()

    while true:
        
        tkpy.predict_price()




       
        
    



main()