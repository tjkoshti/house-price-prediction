import tkinter as tk
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tkinter import ttk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np
from pandas_ods_reader import read_ods 
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests, zipfile, io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,classification_report, confusion_matrix, accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd #Importing Pandas
from matplotlib import pyplot as plt
from PIL import Image,ImageTk
import easygui as eg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from tkinter import PhotoImage
borough_list=pd.DataFrame(pd.read_excel('districts.xls'))
borough_list.rename(columns = {'District':'Borough_name','Ceremonial County':'County'}, inplace = True)
borough_list = borough_list.applymap(lambda s: s.lower() if type(s) == str else s)
import sys
c_b = borough_list.Borough_name.values.tolist()
#values=borough_list['Borough_name'].tolist()
# Load the data
data = pd.read_csv('cleaned_data.csv')
 


# Create the GUI
root = tk.Tk()
# Set the window size to the size of the image
#image = tk.PhotoImage(file='giphy.gif')
#image = image.subsample(2, 2)
# Create the label
#label = tk.Label(root, image=image)

#label.place(x=0, y=0, relwidth=1, relheight=0.2)
#label.pack()


root.title('House Price Prediction')
root.configure(bg="pink")

    
# Create a label and text input for the location

location_label = tk.Label(root, text='Please Enter any Valid Location in England:*',font=('Arial', 20, 'bold'))
location_label.configure(bg="pink")
location_entry = tk.Entry(root)
location_label.pack(padx=10, pady=10)
location_entry.pack(padx=10, pady=10)
"""
location_label = tk.Label(root, text='lease Enter any Valid Location in England*:',font=('Arial', 20, 'bold'))
location_combobox = ttk.Combobox(root, values=['sheffield'])
location_label.pack(padx=10, pady=10)
location_combobox.pack(padx=10, pady=10)
"""
htype_label = tk.Label(root, text='Please select an option for house Type:*\n0 - Detached  1 - Flat 2 - Semi-Detached  3 - Terraced ',font=('Arial', 20, 'bold'))
htype_label.configure(bg="pink")
htype_combobox = ttk.Combobox(root, values=[0,1,2,3])
htype_label.pack(padx=10, pady=10)
htype_combobox.pack(padx=10, pady=10)


build_label = tk.Label(root, text='Please select an option for the build type*:\n 0 - New     1-Old',font=('Arial', 20, 'bold'))
build_label.configure(bg="pink")
build_combobox = ttk.Combobox(root, values=[0,1])
build_label.pack(padx=10, pady=10)
build_combobox.pack(padx=10, pady=10)

lease_label = tk.Label(root, text='Please select an option for the lease type*: \n 0 - Freehold     1-Lease',font=('Arial', 20, 'bold'))
lease_label.configure(bg="pink")
lease_combobox = ttk.Combobox(root, values=[0,1])
lease_label.pack(padx=10, pady=10)
lease_combobox.pack(padx=10, pady=10)

# Create a button to run the prediction
def predict_price():
    #location = eg.enterbox(msg='Enter a location:')
    #data= pd.get_dummies(df) 
    while True:

        location = location_entry.get()
        location = location.lower()
        #location=str(location)

        
        if location in data['Borough_name'].values  :
            # location was found in DataFrame, so exit the loop
            #data_location=data[data['Borough_name'] == location]
            break
        elif  location== '':
            eg.msgbox("Error: Location cannot be empty. Please try again.")
            
            break
            
        else:
            eg.msgbox('Error: location not found in DataFrame. Please try again.\n please select from the given Borough')
            
        break
        
    while True:
        htype = htype_combobox.get()
        htype=int(htype)
        
        if htype in data['house_type'].values:
            
            break


        elif htype =='':
            
            eg.msgbox("Error: House type cannot be empty. Please try again.")
            
            break
            
        else:
            eg.msgbox('Error: Choice Does not Exist!\nPlease Enter a valid house type from mentioned choices \n0 - Detached \n1 - Flat \n2 - Semi-Detached  \n3 - Terraced ')
            
        
        break

    

    while True:   

        build = build_combobox.get()
        build=int(build)
        if build in data['new_build'].values:
            
            break


        elif  build == "":
        
           
            eg.msgbox("Error: Build Type cannot be empty. Please try again.")
            
            break
        else:
            eg.msgbox('Error: Choice Does not Exist!\nPlease Enter a valid Build type from mentioned choices\n0 - New \n1 - Old')
            break
        break
 
    
        
    while True:
        lease = lease_combobox.get()
        lease=int(lease)
        if lease in data['lease_type'].values:
            
            break


        elif lease =='':
            eg.msgbox("Error: Lease Type cannot be empty. Please try again.")
            
            break
        elif lease not in data['lease_type'].values:
            eg.msgbox('Choice Does not Exist!\nPlease Enter a valid Lease type from mentioned choices \n0 - Freehold \n1 - Lease')
            break
        break
   
    
        
    

  # Use the DataFrame to select the rows with the given location
    data_location = data[data['Borough_name'] == location]
    clinics=data_location['clinics_count'].mean()
    schools=data_location['schools_count'].mean()
    Cash_Sales_Volume=data_location['schools_count'].mean()
    Mortgage_Sales_Volume=data_location['Mortgage_Sales_Volume'].mean()
    hospital=data_location['hospital_count'].mean()
    pubs=data_location['pubs_count'].mean()
    gp=data_location['gp_count'].mean()
    Income=data_location['Income Decile'].mean()
    Emplyment=data_location['Emplyment Decile'].mean()
    Crime=data_location['Crime Decile'].mean()
    population=data_location['population'].mean()
    distance_to_park=data_location['schools_count'].mean()
    parks=data_location['avg_number_of_parks'].mean()
    lat=data_location['lat'].mean()
    lng=data_location['lng'].mean()
    #lease=int(lease)
    #build=int(build)
    #htype=int(htype)

    from sklearn.model_selection import KFold
    
    le = LabelEncoder()
    for column in data_location[['Borough_name']].columns:
        data_location[column]=le.fit_transform(data_location[column].values)
    # Train a linear regression model on the filtered data
    X = data_location.drop(['Unnamed: 0','Borough_name','County','price'],axis=1)
    
    y = data_location[['price']]
    kfold = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kfold.split(data_location):
      # Split the data into training and test sets
        X_train, X_test = data_location[[ 'house_type', 'new_build',
           'lease_type', 'Cash_Sales_Volume', 'Mortgage_Sales_Volume',
           'clinics_count', 'schools_count', 'hospital_count', 'pubs_count',
           'gp_count', 'Income Decile', 'Emplyment Decile', 'Crime Decile',
           'population', 'Avg_distance_to_park', 'avg_number_of_parks', 'lat',
           'lng']].iloc[train_index], data_location[[ 'house_type', 'new_build',
           'lease_type', 'Cash_Sales_Volume', 'Mortgage_Sales_Volume',
           'clinics_count', 'schools_count', 'hospital_count', 'pubs_count',
           'gp_count', 'Income Decile', 'Emplyment Decile', 'Crime Decile',
           'population', 'Avg_distance_to_park', 'avg_number_of_parks', 'lat',
           'lng']].iloc[test_index]
        
    Y_train, Y_test = data_location["price"].iloc[train_index], data_location["price"].iloc[test_index]
    
    pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', None)  # Placeholder for the model
    ])

    # Define the k-fold cross-validation
    #kfold = KFold(n_splits=2)
    scores = []
    # Train and evaluate multiple models
    model_accuracies = {}
    models=[RandomForestRegressor(n_estimators=200,random_state=42,max_features=0.2),KNeighborsRegressor(), XGBRegressor()]
    # Loop through a list of models
    for model in models:
        # Set the model in the pipeline
        pipeline.set_params(model=model)

        # Evaluate the model using k-fold cross-validation
        score = cross_val_score(pipeline, X, y, cv=kfold,scoring='r2')
        #accuracy = scores.mean()

        # Store the scores
        scores.append(score)
    

    # Find the index of the model with the highest mean score
    best_index = np.argmax([np.mean(score) for score in scores])
    
    best_model = models[best_index]


    # Evaluate the best model on the test set
    best_model.fit(X_train,Y_train)

    #accuracy = best_model.score(X_test, Y_test)
    # Use the model to predict the price of the house
    price = best_model.predict(np.array([[htype, build, lease, schools, clinics,Cash_Sales_Volume,Mortgage_Sales_Volume,hospital,pubs,gp,Income,Emplyment,Crime,population,distance_to_park,parks,lat,lng]]))[0]
    #print(price)
    ypred_best=best_model.predict(X_test)
    
    #results_window = tk.Toplevel()

    # set the title of the window
    #results_window.title("Results")
    #result_label = tk.Label(results_window, text='')
    #result_label.pack(padx=10, pady=10)
    
    eg.msgbox(f'Predicted Price for {location} is: {price:.2f}'+ "\nNumber of pubs " + str(pubs)+ "\nNumber of clinics " + str(clinics)+ "\nNumber of schools " + str(schools)+ "\nNumber of hospitals " + str(hospital) + "\nNumber of GPs " + str(gp)+ "\nNumber of population " + str(population)+ "\n Avg Number of parks " + str(parks)+ "\n Crime Decile " + str(Crime)+ "\n Income Decile " + str(Income)+ "\n Employement Decile " + str(Emplyment))
        #result_label.config(text=f'Prediction: {price:.0f}£')
    msg = "Do you want to continue?"
    title = "Please Confirm"
    if eg.ccbox(msg, title):     # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:  # user chose Cancel
        sys.exit(0)
    

predict_button = tk.Button(root, text='Predict', command=predict_price)
predict_button.pack()

# Create a label to display the result


# Run the GUI
root.mainloop()