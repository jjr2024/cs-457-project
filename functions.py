import requests
import json 
import pandas as pd
#need to "pip install yfinance"
import yfinance as yahooFinance

def get_tickers():
    """
        Reads through the top 500 company tickers and returns them in an array
    """
    file = open("stock_tickers.txt", "r")
    content = file.read()
    tickers = content.split("\n")
    file.close()

    return tickers

def get_transcript(ticker, quarter, year):
    """
        symbol, quarter, year, date, content
    """
    res = requests.get('https://discountingcashflows.com/api/transcript/'+ticker+'/'+quarter+'/'+year+'/')
    response = json.loads(res.text)
    if len(response) == 0:
        #print("No " + quarter + " " + year + " transcript for "+ticker)
        return response
    else:
        return response

def get_price_data(ticker):
    GetInformation = yahooFinance.Ticker(ticker)
    return GetInformation.history(period="max")

def get_stock_dict():
    tickers = get_tickers()
    stock_dict_df = dict()
    for tick in tickers:
        stock_dict_df[tick] = pd.read_csv('Stock_Prices/'+tick+'.csv')
    return stock_dict_df

def get_companies_by_50():
    f = open('Transcript_json/companies_50.json')
    classes_0_50 = json.load(f)
    f.close()
    f = open('Transcript_json/companies_100.json')
    classes_50_100 = json.load(f)
    f.close()
    f = open('Transcript_json/companies_150.json')
    classes_100_150 = json.load(f)
    f.close()
    f = open('Transcript_json/companies_200.json')
    classes_150_200 = json.load(f)
    f.close()
    f = open('Transcript_json/companies_250.json')
    classes_200_250 = json.load(f)
    f.close()
   
    return classes_0_50, classes_50_100, classes_100_150, classes_150_200, classes_200_250