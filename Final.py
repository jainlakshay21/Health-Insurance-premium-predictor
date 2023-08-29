import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import customtkinter
import tkinter

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

app = customtkinter.CTk()  # create CTk app like you do with the Tk app
app.geometry("1400x1000")

#bg = ImageTk.PhotoImage(Image.open("banner-with-rainbow-coloured-low-poly-design_1048-12754.webp"))
im= Image.open("fmg.jpg")


# Cropped image of above dimension
# (It will not change original image
newsize = (1400, 1000)
im1 = im.resize(newsize)
bg = ImageTk.PhotoImage(im1)
  
# Show image using label
label1 = Label(app, image = bg)
label1.place(x = 0, y = 0)

title = app.title("HEALTH INSURANCE PREMIUM PREDICTOR")


label = customtkinter.CTkLabel(master=app, text="HEALTH INSURANCE\nPREMIUM PREDICTOR")
label.config(font=('Helvetica bold',20))
label.place(relx=0.5, rely=0.08, anchor=tkinter.CENTER)

#name
label_name = customtkinter.CTkLabel(master=app, text="Your Name")
label_name.config(font=('Helvetica bold',10))
label_name.place(relx=0.3, rely=0.17, anchor=tkinter.N)

#textbox = customtkinter.CTkTextbox(app)
#textbox.place(relx = 0.3, rely = 0.2, anchor = tkinter.N)

#textbox.insert("0.0", "new text to insert")  # insert at line 0 character 0
#text = textbox.get("0.0", "end")  # get text from line 0 character 0 till the end

Name = customtkinter.CTkEntry(master = app, width=350)
Name.place(relx = 0.5, rely = 0.187, anchor = tkinter.CENTER)



#sex
label_sex = customtkinter.CTkLabel(master=app, text="Your Sex")
label_sex.config(font=('Helvetica bold',10))
label_sex.place(relx=0.3, rely=0.27, anchor=tkinter.N)

def optionmenu_callback(choice):
    print("optionmenu dropdown clicked:", choice)

clicked1 = customtkinter.CTkOptionMenu(master=app,
                                       values=["Male","Female", "Other"],
                                       command=optionmenu_callback)
clicked1.pack(padx=20, pady=213)
clicked1.set("Male")  # set initial value



#age
label_age = customtkinter.CTkLabel(master=app, text="Your Age")
label_age.config(font=('Helvetica bold',10))
label_age.place(relx=0.3, rely=0.37, anchor=tkinter.N)

Age = customtkinter.CTkEntry(master = app, width=350)
Age.place(relx = 0.5, rely = 0.388, anchor = tkinter.CENTER)



#bmi 
label_bmi = customtkinter.CTkLabel(master=app, text="Your BMI")
label_bmi.config(font=('Helvetica bold',10))
label_bmi.place(relx=0.3, rely=0.47, anchor=tkinter.N)

BMI = customtkinter.CTkEntry(master = app, width=350)
BMI.place(relx = 0.5, rely = 0.488, anchor = tkinter.CENTER)



#children 
#bmi 
label_child = customtkinter.CTkLabel(master=app, text="Enter number of children")
label_child.config(font=('Helvetica bold',10))
label_child.place(relx=0.3, rely=0.57, anchor=tkinter.N)

NC = customtkinter.CTkEntry(master = app, width=350)
NC.place(relx = 0.5, rely = 0.588, anchor = tkinter.CENTER)



#smoker
label_smoker = customtkinter.CTkLabel(master=app, text="Are you a smoker")
label_smoker.config(font=('Helvetica bold',10))
label_smoker.place(relx=0.3, rely=0.67, anchor=tkinter.N)

def optionmenu_callback1(f):
    print("optionmenu dropdown clicked:", f)

clicked2 = customtkinter.CTkOptionMenu(master=app,
                                       values=["Yes","No"],
                                       command=optionmenu_callback1)
clicked2.pack(padx=20, pady=72.5)
clicked2.set("Yes")  # set initial value



#region
label_region = customtkinter.CTkLabel(master=app, text="Enter your region")
label_region.config(font=('Helvetica bold',10))
label_region.place(relx=0.3, rely=0.77, anchor=tkinter.N)

def optionmenu_callback2(g):
    print("optionmenu dropdown clicked:", g)

clicked3 = customtkinter.CTkOptionMenu(master=app,
                                       values=["South East","South West","North East","North West"],
                                       command=optionmenu_callback2)
clicked3.place(relx = 0.5, rely = 0.786, anchor = tkinter.CENTER)
clicked3.set("South East")  # set initial value


#email
label_email = customtkinter.CTkLabel(master=app, text="Enter your email address")
label_email.config(font=('Helvetica bold',10))
label_email.place(relx=0.3, rely=0.87, anchor=tkinter.N)

Email = customtkinter.CTkEntry(master = app, width=350)
Email.place(relx = 0.5, rely = 0.887, anchor = tkinter.CENTER)



#submit
def button_event():

    data = pd.read_csv('insurance.csv')
    data.head()
    data.info()
    ### There are no missing values as such
    data['region'].value_counts().sort_values()
    data['children'].value_counts().sort_values()
    ### Converting Categorical Features to Numerical
    clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                    'smoker': {'no': 0 , 'yes' : 1},
                    'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
                }
    data_copy = data.copy()
    data_copy.replace(clean_data, inplace=True)
    data_copy.describe()
    ### Plotting Skew and Kurtosis
    print('Printing Skewness and Kurtosis for all columns')
    print()
    for col in list(data_copy.columns):
        print('{0} : Skewness {1:.3f} and  Kurtosis {2:.3f}'.format(col,data_copy[col].skew(),data_copy[col].kurt()))

    ### There might be few outliers in Charges but then we cannot say that the value is an outlier as there might be cases in which Charge for medical was very les actually!
    ### Prepating data - We can scale BMI and Charges Column before proceeding with Prediction
    from sklearn.preprocessing import StandardScaler
    data_pre = data_copy.copy()

    tempBmi = data_pre.bmi
    tempBmi = tempBmi.values.reshape(-1,1)
    data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)

    tempAge = data_pre.age
    tempAge = tempAge.values.reshape(-1,1)
    data_pre['age'] = StandardScaler().fit_transform(tempAge)

    tempCharges = data_pre.charges
    tempCharges = tempCharges.values.reshape(-1,1)
    data_pre['charges'] = StandardScaler().fit_transform(tempCharges)

    data_pre.head()
    X = data_pre.drop('charges',axis=1).values
    y = data_pre['charges'].values.reshape(-1,1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    print('Size of X_train : ', X_train.shape)
    print('Size of y_train : ', y_train.shape)
    print('Size of X_test : ', X_test.shape)
    print('Size of Y_test : ', y_test.shape)
    ## Importing Libraries
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    import xgboost as xgb

    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV


    ## Training Data without Scaling for RandomClassifier
    data_copy.head()
    X_ = data_copy.drop('charges',axis=1).values
    y_ = data_copy['charges'].values.reshape(-1,1)

    from sklearn.model_selection import train_test_split
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_,y_,test_size=0.2, random_state=42)

    print('Size of X_train_ : ', X_train_.shape)
    print('Size of y_train_ : ', y_train_.shape)
    print('Size of X_test_ : ', X_test_.shape)
    print('Size of Y_test_ : ', y_test_.shape)
    rf_reg = RandomForestRegressor(max_depth=50, min_samples_leaf=12, min_samples_split=7,
                        n_estimators=1200)
    rf_reg.fit(X_train_, y_train_.ravel())
    y_pred_rf_train_ = rf_reg.predict(X_train_)
    r2_score_rf_train_ = r2_score(y_train_, y_pred_rf_train_)

    y_pred_rf_test_ = rf_reg.predict(X_test_)
    r2_score_rf_test_ = r2_score(y_test_, y_pred_rf_test_)

    print('R2 score (train) : {0:.3f}'.format(r2_score_rf_train_))
    print('R2 score (test) : {0:.3f}'.format(r2_score_rf_test_))
    import pickle

    Pkl_Filename = "rf_tuned.pkl"  

    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(rf_reg, file)
    # Load the Model back from file
    with open(Pkl_Filename, 'rb') as file:  
        rf_tuned_loaded = pickle.load(file)
    rf_tuned_loaded
    
    if (clicked1.get()=='Male'):
        sex=0
    else:
        sex=1
    
    if (clicked2.get()=='No'):
        smoker=0
    else:
        smoker=1
    
    if (clicked3.get()=='South East'):
        region=2
    elif(clicked3.get()=='South West'):
        region=3
    elif(clicked3.get()=='North East'):
        region=1
    elif(clicked3.get()=='North West'):
        region=0
    
    pred=rf_tuned_loaded.predict(np.array([int(Age.get()),sex,float(BMI.get()),int(NC.get()),smoker,region]).reshape(1,6))[0]
    print('{0:.3f}'.format(pred))
    #input_data = (int(Age.get()),sex,float(BMI.get()),int(NC.get()),smoker,region)

    
    Submit_label=Label(app,text='Thank you '+ Name.get() +', Your insurance cost is USD '+ str(abs(round(pred))))
    Submit_label.place(relx=0.7, rely=0.93, anchor=tkinter.N)

    Submit_label1=Label(app,text='Your Email has been sent.')
    Submit_label1.place(relx=0.7, rely=0.96, anchor=tkinter.N)
    
    
    def send_mail():

        body = 'Hello {},\n\nYour info:-\n     Age: {}\n     BMI: {}\n     Sex: {}\n     No of children: {}\n     Are you a smoker: {}\n     Region: {}\n\nAs per the details provided above by you, the approximate insurance amount will be around $ {}.'.format(Name.get(),Age.get(),BMI.get(),clicked1.get(),NC.get(),clicked2.get(),clicked3.get(),round(pred))

        sender = "kushwork1206@gmail.com"

        password = 'aquuteqiuqbjtdca'
        # put the email of the receiver here
        receiver = Email.get()

        #Setup the MIME
        message = MIMEMultipart()
        message['From'] = "kushwork1206@gmail.com"
        message['To'] = Email.get()
        message['Subject'] = 'Quote for insurance policy'

        message.attach(MIMEText(body, 'plain'))

        pdfname = 'Report.pdf'

        # open the file in bynary
        binary_pdf = open(pdfname, 'rb')

        payload = MIMEBase('application', 'octate-stream', Name=pdfname)
        # payload = MIMEBase('application', 'pdf', Name=pdfname)
        payload.set_payload((binary_pdf).read())

        # enconding the binary into base64
        encoders.encode_base64(payload)

        # add header with pdf name
        payload.add_header('Content-Decomposition', 'attachment', filename=pdfname)
        message.attach(payload)

        #use gmail with port
        session = smtplib.SMTP('smtp.gmail.com', 587)

        #enable security
        session.starttls()

        #login with mail_id and password
        session.login(sender, password)

        text = message.as_string()
        session.sendmail(sender, receiver, text)
        session.quit()
        print('Mail Sent')    
    send_mail()
button1 = customtkinter.CTkButton(master=app,
                                 width=200,
                                 height=42,
                                 border_width=0,
                                 corner_radius=8,
                                 text="Submit",
                                 command=button_event)
button1.place(relx=0.5, rely=0.960, anchor=tkinter.CENTER)
app.mainloop()
