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
    return response[0]

def get_price_data(ticker):
    GetInformation = yahooFinance.Ticker(ticker)
    return GetInformation.history(period="max")

def get_stock_dict():
    tickers = get_tickers()
    stock_dict_df = dict()
    for tick in tickers:
        stock_dict_df[tick] = pd.read_csv('Stock_Prices/'+tick+'.csv')
    return stock_dict_df
