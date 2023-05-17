import pandas as pd
import math
import numpy as np

from statsmodels.base.model import GenericLikelihoodModelResults
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]  # ist prei in tausend
df_data = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B','LSTAT'])


data = df_data

features = data.drop(["INDUS","AGE"], axis =1)

log_prices = np.log(target)
#machen log_prices 2 dimensional
target = pd.DataFrame(log_prices, columns = ["PRICE"])

# feauters hat shape (506,11), das brauchen wir auch für unser Target
# unser Target ist momentan 506,1 
property_stats  = np.ndarray(shape=(1, 11)) # erstellt ein leeres array mit nullen


property_stats = features.mean().values.reshape(1,11)  # wir befüllen es jetzt und formatierten in ein array

property_stats


regr = LinearRegression().fit(features, target) # allle thetas berechenen
fitted_vals = regr.predict(features)
#fitted_vals.shape
#target.shape

#mse und rmse berechnen

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)
#features["RM"] = round(features["RM"],0)
features.head()

# python funktion welche log house prices schätzen wird

def get_log_estimate(nr_rooms, students, river = False,
                     high_confidence = True):
    
   
    if nr_rooms <1 or students <1:
        print("Unrealistic values, try again!")
    else:
        property_stats[0][4] = nr_rooms

        property_stats[0][8] = students

        if river:
             property_stats[0][2] = 1
        else: 
             property_stats[0][2] = 0

        # gibt uns property steps, welches die durchschnittswerte für alle features
        #hat außer die die wir in der Funktion ändern
        log_estimate = regr.predict(property_stats)[0][0]

        # jetzt wollen wir noch eine 95% kondifente aussage treffen können
        if high_confidence:
            upper_bound = log_estimate + 2*RMSE
            lower_bound = log_estimate - 2*RMSE
            intervall = 95
        else:
            upper_bound = log_estimate + RMSE
            lower_bound = log_estimate - RMSE
            intervall = 68


        # log price in dollar umwandeln
     #log_estimate  = math.ceil((np.e**log_estimate) * 7.82 *1000)
      #  upper_bound = math.ceil((np.e**upper_bound)*7.82 *1000)
       # lower_bound = math.ceil((np.e**lower_bound) * 7.82 *1000)'''
        return log_estimate, upper_bound, lower_bound, intervall


#print(get_log_estimate(3,20, river = True, high_confidence = False))
#print(get_log_estimate(3,20, river = True ))

def get_dollar_est(rm, ptratio, chas = False, large_range = True):
    """Estimate the price in dollar, takes 4 inputs 
    Keyword arguments: 
    rm -- roomnumer 
    ptratio -- amount of students per teacher 
    chas -- True if close to the river, 
    large_range -- True for 95% prediction Intervall; False for 68%
    """
    log_est,upper,lower, conf = get_log_estimate(nr_rooms = rm, students = ptratio, river = chas, high_confidence = large_range)

    dollar_est = math.ceil((np.e**log_est) * 7.82 *1000)
    dollar_up = math.ceil((np.e**upper)*7.82 *1000)
    dollar_lo = math.ceil((np.e**lower) * 7.82 *1000)

    print(f"der heutige Dollarpreis liegt bei {dollar_est}$ und mit {conf}% in einem Intverall von {dollar_up}$ : {dollar_lo}$")
    
#get_dollar_est(4, 10)

