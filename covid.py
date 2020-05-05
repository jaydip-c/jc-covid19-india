import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

def convertDigit(txt):
    if txt.replace(",", '').isdigit():
        return int(txt.replace(',', ''))
    return txt

class Covid:
    
    def predict_india(self, url1, url2):
        idf = self.get_india_data(url1, url2)

        cases = idf['confirmed'].groupby(idf['date']).sum().sort_values(ascending=True)
        cases = cases[cases>0].reset_index().drop('date',axis=1)
        
        deaths = idf['deaths'].groupby(idf['date']).sum().sort_values(ascending=True)
        deaths = deaths[deaths>0].reset_index().drop('date',axis=1)
        
        days_since_first_case = np.array([i for i in range(len(cases.index))]).reshape(-1, 1)
        icases = np.array(cases).reshape(-1, 1)
        
        days_since_first_death = np.array([i for i in range(len(deaths.index))]).reshape(-1, 1)
        ideaths = np.array(deaths).reshape(-1, 1)
        
        days_in_future = 5
        future_forcast = np.array([i for i in range(len(cases.index)+days_in_future)]).reshape(-1, 1)
        adjusted_dates = future_forcast[:-days_in_future]
        future_forcast_deaths = np.array([i for i in range(len(deaths.index)+days_in_future)]).reshape(-1, 1)
        adjusted_dates_deaths = future_forcast_deaths[:-days_in_future]
        
        X_train, X_test, y_train, y_test = train_test_split(days_since_first_case
                                                        , cases
                                                        , test_size= 5
                                                        , shuffle=False
                                                        , random_state = 42) 
    
        X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_first_death
                                                        , deaths
                                                        , test_size= 5
                                                        , shuffle=False
                                                        , random_state = 42)
        
        # looking for best degree for deaths
        mae = 10000
        degree = 0
        for i in range(101):
            # Transform our cases data for polynomial regression
            poly = PolynomialFeatures(degree=i)
            poly_X_train = poly.fit_transform(X_train)
            poly_X_test = poly.fit_transform(X_test)
            poly_future_forcast = poly.fit_transform(future_forcast)
        
            # polynomial regression cases
            linear_model = LinearRegression(normalize=True, fit_intercept=False)
            linear_model.fit(poly_X_train, y_train)
            test_linear_pred = linear_model.predict(poly_X_test)
            linear_pred = linear_model.predict(poly_future_forcast)
        
            # evaluating with MAE and MSE
            m = mean_absolute_error(test_linear_pred, y_test)
            if(m<mae):
                mae = m
                degree = i
            if(i==100):
                pass
#                print('the best mae for cases is:',round(mae,2))
#                print('the best degree for cases is:',degree)
    
        # looking for best degree for deaths
        mae = 10000
        degree = 0
        for i in range(101):
            # Transform our death data for polynomial regression
            poly_death = PolynomialFeatures(degree=i)
            poly_X_train_death = poly_death.fit_transform(X_train_death)
            poly_X_test_death = poly_death.fit_transform(X_test_death)
            poly_future_forcast_death = poly_death.fit_transform(future_forcast_deaths)
        
            # polynomial regression deaths
            linear_model_death = LinearRegression(normalize=True, fit_intercept=False)
            linear_model_death.fit(poly_X_train_death, y_train_death)
            test_linear_pred_death = linear_model_death.predict(poly_X_test_death)
            linear_pred_death = linear_model_death.predict(poly_future_forcast_death)
        
            # evaluating with MAE and MSE
            m = mean_absolute_error(test_linear_pred_death, y_test_death)
            if(m<mae):
                mae = m
                degree = i
            if(i==100):
                pass
#                print('the best mae for death is:',round(mae,2))
#                print('the best degree for deaths is:',degree)
                
        # Transform our cases data for polynomial regression
        poly = PolynomialFeatures(degree=3)
        poly_X_train = poly.fit_transform(X_train)
        poly_X_test = poly.fit_transform(X_test)
        poly_future_forcast = poly.fit_transform(future_forcast)
        
        # Transform our death data for polynomial regression
        poly_death = PolynomialFeatures(degree=5)
        poly_X_train_death = poly_death.fit_transform(X_train_death)
        poly_X_test_death = poly_death.fit_transform(X_test_death)
        poly_future_forcast_death = poly_death.fit_transform(future_forcast_deaths)
        
        # polynomial regression cases
        linear_model = LinearRegression(normalize=True, fit_intercept=False)
        linear_model.fit(poly_X_train, y_train)
        test_linear_pred = linear_model.predict(poly_X_test)
        linear_pred = linear_model.predict(poly_future_forcast)
        
        # evaluating with MAE and MSE
#        print('MAE:', mean_absolute_error(test_linear_pred, y_test))

        p = linear_pred[len(cases):]
        d = [(datetime.today() + timedelta(days = i)).strftime("%Y-%m-%d") for i in range(len(p))]
        a = pd.DataFrame(d, columns = ['Dates'])
        b = pd.DataFrame(p, columns = ['Predicted'])
        pdf = pd.concat([a, b], axis=1)
    
        return (adjusted_dates, icases, future_forcast, linear_pred, pdf)
        
    def get_world_data(self, url):
        self.url = url
        
        resp = requests.get(self.url)
        soup = bs(resp.content, 'lxml')
        table = soup.findAll("table", {'class':'main_table_countries'})[0]
        rows = table.findAll('tr')

        wdata = []
        try:
            for row in rows:
                data = []
                for cell in row.findAll(['td', 'th']):
                    data.append(convertDigit(cell.text.strip()))
                wdata.append(data)
        finally:
                pass
    
        column_names = wdata.pop(0)
        df = pd.DataFrame(wdata, columns=column_names)
        df.rename(columns = {'Country,Other':'Country'}, inplace=True)
        new_df = df[8:18]
    
        tdf = new_df[new_df['Country'].isin(['India'])]
        if tdf.empty:
            tdf = df[df['Country'].isin(['India'])]
            new_df = new_df.append(tdf, ignore_index=True)
        tot = df[-1:]
        new_df = new_df.append(tot, ignore_index=True)
        new_df = new_df[['Country', 'TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases']]
        new_df.fillna(0)

        return new_df

    def get_india_data(self, url1, url2):
        r = requests.get(url2)
        data = r.json()
        india = data["India"]
        india_data = [d for d in india if d.get("confirmed") > 0]
        
        df1 = pd.DataFrame(india_data)
        df1['active'] = df1.confirmed - df1.deaths - df1.recovered
        
        country = 'India'
        
        resp = requests.get(url1)
        soup = bs(resp.content, "lxml")
        table = soup.findAll("table", {'class':'main_table_countries'})[0]
        tr_elems = table.find_all("tr") # All rows in table
        data = []
        for tr in tr_elems: # Loop through rows
            td_elems = tr.find_all("td") # columns in row
            for td in td_elems:
                if td.text.strip() == country:
        
                    data.append([convertDigit(td1.text.strip()) for td1 in td_elems])
        
        d = {}
        d['confirmed'] = data[0][1]
        d['deaths'] = data[0][3]
        d['recovered'] = data[0][5]
        d['active'] = data[0][6]
        d['date'] = datetime.today().strftime('%Y-%m-%d')
        
        tdf = pd.DataFrame(pd.Series(d)).T
        ddf = df1.append(tdf, ignore_index=True, sort=True)
        
        cols = ddf.columns.tolist()
        cols = ['confirmed', 'active', 'deaths', 'recovered', 'date']
        ddf = ddf[cols]
        
        return ddf
