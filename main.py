# libraries and module 

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from flask_cors import CORS
from flask import Flask
from flask import request
import pandas as pd
import re
from flask_pymongo import pymongo
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
from datetime import date
import json
import numpy as np
import hashlib
from werkzeug.utils import secure_filename
import os

# global variables 
df = pd.DataFrame()
filename = ''
rmse = ''
accuracy = ''
MAPE = ''

# flask establishment
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './static'

# file upload allowed extension
ALLOWED_EXTENSIONS = {'csv'}

# regex for email validation
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# database connection establishment

con_string = "mongodb+srv://kiran:kiran@cluster0.9bnuchr.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(con_string)

db = client.get_database('example')

user_collection = pymongo.collection.Collection(db, 'collectionsexample') #(<database_name>,"<collection_name>")
print("MongoDB connected Successfully")


# routings

@app.route('/')
def welcome():
    return 'welcome'


# user signup
@app.route('/register-user', methods=['POST'])
def register_user():
    msg=''
    try:
        req_body = request.get_json(force=True)
        var=req_body['username']
        if re.fullmatch(regex,req_body['username']):
            if (not user_collection.find_one({"username":var})):
                # Hashing password using message digest5 algorithm
                password=req_body['password']
                hash_password=hashlib.md5(password.encode()).hexdigest()

                # username and passwords are inserted to mongoDB using insert_one function
                user_collection.insert_one({"username":var,"password":hash_password}) 
                msg='SignUp Successful'
            else:
                msg='User Already Exists'
        else:
            msg='User Name is not an email'
    
    except Exception as e:
        print(e)
        msg='User Already Exists'
    return {'resp': msg}

# user signin
@app.route('/signin',methods=['POST'])
def signin():
    msg=''
    try:
        data=request.get_json(force=True)
        print(data)
        var=data['username'] 
       
        # Hashing password using message digest5 algorithm
        password=data['password']
        hash_password=hashlib.md5(password.encode()).hexdigest()

        # username and password are comapred with mongoDB using find_one function
        out=user_collection.find_one({"username":var,"password":hash_password})
        u1=out.get('username')
        p1=out.get('password')
        
        msg='Login Successful'
    except Exception as e:
        print(e)
        msg='Unsuccessful'
    return {'resp': msg}

# allowed file function
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# file upload

@app.route('/file_upload', methods=['GET', 'POST'])
def fileUpload():
    msg = ''
    if request.method == 'POST':

        # Checking whether File is available in the request
        if 'file' not in request.files:

            msg = 'File not attached'

        file = request.files.get('file')
        
        # Checking whether empty filename is sent in request
        if file.filename == '':
            msg = 'Please select a file'

        # If file is in request and satisfies .csv extension it is made to save
        if file and allowed_file(file.filename):
            global filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            msg = 'file upload success'
        # If file extension is not in .csv file is not saved
        else:
            msg = 'select a csv file'
    return {'response': msg}



# model training on number of days basis

@app.route('/pred_days',methods=['POST'])
def calc_days():
  data=request.get_json(force=True)
  days=int(data['days'])
  global filename
  path = './static/'+filename

  # uploaded csv are read as dataframes and null values are dropped
  df = pd.read_csv(path, parse_dates=True, index_col='Date')
  df = df.dropna()
  
  # SARIMA Seasonal Arima  model
  # SARIMA is similar to ARIMA but executes both seasonal and stationary range of values
  # SARIMAX Algorithm comprises functionalities of all ARIMA , SARIMA , ARIMAX and  SARIMAX
  #Among that we chose SARIMA model
  # Sales value is trained in model with respect to Dates which is index column
  model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

  # Model fitting is a measure of how well a machine learning model generalizes to Similar data to that on which it was trained. 
  # A model that is well-fitted produces more accurate outcomes.
  results = model.fit()

  # Now testing is made over the model for prediction. The second half of the dataframes is tested.
  # The comparison between actual and tested second half of graph values is Actual Vs Predicted Sales
  df['forecast'] = results.predict(start=50, end=103, dynamic=True)

  # Values are stored as numpy array and converted to list.
  # The lis is converted to JSON

  predicted_sales1 = df['forecast'][50:103].to_numpy()
  predicted_sales2 = predicted_sales1.tolist()
  predicted_sales = json.dumps(predicted_sales2)

  actual_sales1 = df['Sales'][50:103].to_numpy()
  actual_sales2 = actual_sales1.tolist()
  actual_sales = json.dumps(actual_sales2)

  dates1 = list(df.index[50:103])
  dates2 = [str(date)[:-9] for date in dates1]
  dates = json.dumps(dates2)

  # Forecasted Sales
  # Dates are read as time stamps. Sliced to get in the form of YYYY/MM/DD

  future_sales = []
  final = str(df.index[-1])[:-9].split('-')
  final_date = date(int(final[0]), int(final[1]), int(final[2]))
  extra_days = days
  start = 0
  end = extra_days

  # Dataframe is extended upto required date falues. 
  # Future Dates are concatenated to the existing dataframe
  future_dates = [df.index[-1]+DateOffset(days=x) for x in range(0, extra_days+1)]
  future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
  future_df = pd.concat([df, future_date_df])
  future_df['forecast'] = results.predict(start=start+105, end=105+end+1, dynamic=False)
  
  # Numpy array to JSON conversion 
  future_sales1 = future_df['forecast'][105+start:].to_numpy()
  future_sales2 = future_sales1.tolist()
  future_sales = json.dumps(future_sales2)

  future_user_date1 = list(future_df.index[-days:])
  future_user_date2 = [str(date)[:-9] for date in future_user_date1]
  future_user_date = json.dumps(future_user_date2)

  # rmse (ROOT MEAN SQUARE ERROR) calculation
  # Difference between each fields in columns are obtained and obtained as single row
  # Now the difference values are squared and mean of the column is obtained. Whic h is MSE
  # The square root of mse is rmse
  mse = np.square(np.subtract(df['Sales'], df['forecast'])).mean()
  rmse_unparsed = np.sqrt(mse)
  global rmse
  rmse = json.dumps(rmse_unparsed)


  # Accuracy calculation 
  # r2_score compares the dataframe columns and provides results based on comparison
  # Which is multiplied with 100 to get accuracy
  accuracy_unparsed = r2_score(df.Sales[70:103], df.forecast[70:103])
  accuracy_unparsed = str(accuracy_unparsed*100)[:5]+'%'
  global accuracy
  accuracy = json.dumps(accuracy_unparsed)
 

  # MAPE (MEAN ABSOLUTE PERCENTAGE ERROR) calculation
  # measures accuracy of a forecast system. It measures this accuracy as a percentage, 
  # and can be calculated as the average absolute percent error for each time period minus actual values divided by actual values.
  mape_unparsed = mean_absolute_percentage_error(
      df.Sales[70:103], df.forecast[70:103])
  global MAPE
  MAPE = json.dumps(mape_unparsed)

  return {'actual': actual_sales,'predicted': predicted_sales, 'dates': dates, 'future_user_date': future_user_date, 'future_sales': future_sales,'mape':MAPE[:7],'rmse':rmse[:7],'accuracy':accuracy[1:5]}



  
# model training on the basis of from and to date

@app.route('/user_input', methods=['POST'])
def train_data():

    # User input is obtained 
    data = request.get_json(force=True)
    fromm = data['from']
    to = data['to']

    # Reading File Name

    path = './static/'+'jessy_-_Copy.csv'
    df = pd.read_csv(path, parse_dates=True, index_col='Date')

    # NaN values are dropped
    df = df.dropna()

    # Training and fitting of SARIMA model
    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    #Actual Vs Predicted Sales Value
    df['forecast'] = results.predict(start=50, end=103, dynamic=True)

    # Conversion of numpy array to JSON
    predicted_sales1 = df['forecast'][50:103].to_numpy()
    predicted_sales2 = predicted_sales1.tolist()
    predicted_sales = json.dumps(predicted_sales2)

    actual_sales1 = df['Sales'][50:103].to_numpy()
    actual_sales2 = actual_sales1.tolist()
    actual_sales = json.dumps(actual_sales2)

    dates1 = list(df.index[50:103])
    dates2 = [str(date)[:-9] for date in dates1]
    dates = json.dumps(dates2)

    # Forecasted Sales
    # Dates are read as time stamps. Sliced to get in the form of YYYY/MM/DD

    future_sales = []
    final = str(df.index[-1])[:-9].split('-')
    final_date = date(int(final[0]), int(final[1]), int(final[2]))
    date_dict = {}

    # Calculating difference in dates or future prediction and storing it with respective keys in dictionary
    date_dict['from'] = (fromm.split('-'))
    date_dict['to'] = (to.split('-'))

    # Data are read from dictionary with respective index
    from_date = date(int(date_dict['from'][0]), int(
        date_dict['from'][1]), int(date_dict['from'][2]))
    to_date = date(int(date_dict['to'][0]), int(
        date_dict['to'][1]), int(date_dict['to'][2]))
    
    # .days is used to consider the values as date format for finding difference
    difference = (to_date-from_date).days
    extra_days = (to_date-final_date).days
    start = extra_days-difference
    end = extra_days

    # Adding extra days to dataframes
    # future_dates stores the required series of dates
    future_dates = [df.index[-1]+DateOffset(days=x)
                    for x in range(0, extra_days+1)]
    future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    # Future dates are concatenated to dataframe
    future_df = pd.concat([df, future_date_df])
    future_df['forecast'] = results.predict(
        start=start+105, end=105+end+1, dynamic=False)
    

    # Conversion of numpy array to JSON
    future_sales1 = future_df['forecast'][105+start:].to_numpy()
    future_sales2 = future_sales1.tolist()
    future_sales = json.dumps(future_sales2)

    future_user_date1 = list(future_df.index[-difference-1:])
    future_user_date2 = [str(date)[:-9] for date in future_user_date1]
    future_user_date = json.dumps(future_user_date2)

    #  rmse calculation
    mse = np.square(np.subtract(df['Sales'], df['forecast'])).mean()
    rmse_unparsed = np.sqrt(mse)
    global rmse
    rmse = json.dumps(rmse_unparsed)
  

    # accuracy calculation
    accuracy_unparsed = r2_score(df.Sales[70:103], df.forecast[70:103])
    accuracy_unparsed = str(accuracy_unparsed*100)[:5]+'%'
    global accuracy
    accuracy = json.dumps(accuracy_unparsed)
  

    # MAPE calculation
    mape_unparsed = mean_absolute_percentage_error(
        df.Sales[70:103], df.forecast[70:103])
    global MAPE
    MAPE = json.dumps(mape_unparsed)

    resp = {'actual': actual_sales,
            'predicted': predicted_sales, 'dates': dates, 'future_user_date': future_user_date, 'future_sales': future_sales,'mape':MAPE[:7],'rmse':rmse[:7],'accuracy':accuracy[1:5]}

    return resp


# specific date sale prediction

@app.route('/custom_value', methods=['POST'])
def custom_prediction():
    data = request.get_json(force=True)
    custom_unparsed_date = data['custom_date']

    global filename
    path = './static/'+filename
    df = pd.read_csv(path, parse_dates=True, index_col='Date')
    df = df.dropna()
    # SARIMA Model training
    model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    final = str(df.index[-1])[:-9].split('-')
    final_date = date(int(final[0]), int(final[1]), int(final[2]))

    # Parsing into a date
    date_dict = {}
    date_dict['date'] = custom_unparsed_date.split('-')
    # Finding length between last date to custom date
    custom_parsed_date = date(int(date_dict['date'][0]), int(
        date_dict['date'][1]), int(date_dict['date'][2]))
    extra_days = (custom_parsed_date-final_date).days

    # Adding the extra date
    future_dates = [df.index[-1]+DateOffset(days=x)
                    for x in range(0, extra_days+1)]
    future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_date_df])
    future_df['forecast'] = results.predict(
        start=105, end=105+extra_days+1, dynamic=False)
    predicted_non_json_value = future_df['forecast'][-1]
    predicted_value = json.dumps(predicted_non_json_value)

    # Entire testing and training process are performed and the required sales corresponding to specific date is passed as response

    return {'custom_value': predicted_value[0:6]}




if __name__ == '__main__':
    app.run(debug=True)


