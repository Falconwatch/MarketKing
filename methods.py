import yfinance as yf


def load_data(ticker, period_start=None, period_end=None, step='1d'):
    '''
    Returns the historical trade data.

            Parameters:
                    ticker (str): Stock ticker name
                    period_start (string): period start date in format "YYYY-MM-DD". If None - first date in source.
                    period_end (string): period end date in format "YYYY-MM-DD" If None - last date in source.
                    step (str): target step of values. valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

            Returns:
                    stock_df (Pandas DataFrame): DataFrame with ticker stock data of target period
    '''

    assert step in ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], "Wrong step value. should be in ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']"

    stock_df = yf.download(
        tickers = ticker,
        start = period_start,
        end = period_end,
        interval = step,
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
    )
    return stock_df