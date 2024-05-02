import requests
import json 
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
