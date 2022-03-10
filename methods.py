import numpy as np
import pandas as pd
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
    
    assert step in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo',
                    '3mo'], "Wrong step value. should be in ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']"

    stock_df = yf.download(
        tickers=ticker,
        start=period_start,
        end=period_end,
        interval=step,
        group_by='ticker',
        auto_adjust=True,
        prepost=True,
    )
    return stock_df


def mark_target(prices, window=10, how="high"):
    """Размечает фактические лоу и хай точки цены

      Keyword arguments:
      prices -- временной ряд цен (Series)
      window -- окно поиска цен - на сколько впредёд не будет пиков/просадок
      how -- что искать: пики (high) или просадки (low)
      """
    if how == "high":
        facts = prices.rolling(window=window, center=True).max()
    elif how == "low":
        facts = prices.rolling(window=window, center=True).min()

    targets = np.zeros_like(prices)
    targets[prices == facts] = 1
    return targets


def mark_data_targets(data, window=5):
    lows = mark_target(data["Close"], window=window, how="low")
    data["lows"] = lows
    peaks = mark_target(data["Close"], window=window, how="high")
    data["peaks"] = peaks


def create_ma(prices, ma_steps=[5, 10, 15, 20]):
    """Возвращает датафрейм с ценой и её скользящими средними заданной ширины

      Keyword arguments:
      prices -- временной ряд цен (Series)
      ma_steps -- список шагов для скользящего среднего
      """
    result = pd.DataFrame()
    result["price"] = prices
    for ma_step in ma_steps:
        ma = prices.rolling(ma_step).mean()
        result["ma_{}".format(ma_step)] = ma
    return result


def find_sell_signals(short_ma, long_ma):
    """Ищет сигнал "продавать" - короткое МА пересекает длинное МА сверху

    Keyword arguments:
      short_ma -- короткое скользящее среднее (Series)
      long_ma -- длинное скользящее среднее (Series)
    """
    short_higher = short_ma.shift(-1) > long_ma.shift(-1)  # короткое было выше
    short_lower = short_ma.shift(1) < long_ma.shift(1)  # короткое стало ниже

    sell_signals = short_higher & short_lower
    return sell_signals


def find_buy_signals(short_ma, long_ma):
    """Ищет сигнал "покупать" - короткое МА пересекает длинное МА снизу

    Keyword arguments:
      short_ma -- короткое скользящее среднее (Series)
      long_ma -- длинное скользящее среднее (Series)
    """
    short_lower = short_ma.shift(-1) < long_ma.shift(-1)  # короткое было выше
    short_higher = short_ma.shift(1) > long_ma.shift(1)  # короткое стало ниже

    buy_signals = short_higher & short_lower
    return buy_signals


def form_signals(data_ma):
    """Формирует сигналы из множества СС. Важно, чтобы СС в датафрейме шли по возрастанию окна

    Keyword arguments:
      data_ma -- ДатаФрейм с СС разной длины
    """

    sell_signals = pd.DataFrame()
    buy_signals = pd.DataFrame()

    for i in range(1, len(data_ma.columns)):
        short_ma_name = data_ma.columns[i]
        short_ma = data_ma[short_ma_name]
        for j in range(i, len(data_ma.columns)):
            long_ma_name = data_ma.columns[j]
            long_ma = data_ma[long_ma_name]

            sell_signals_name = "{0}_{1}_sell_signal".format(short_ma_name, long_ma_name)
            sell_signals[sell_signals_name] = find_sell_signals(short_ma, long_ma)

            buy_signals_name = "{0}_{1}_buy_signal".format(short_ma_name, long_ma_name)
            buy_signals[buy_signals_name] = find_buy_signals(short_ma, long_ma)

    sell_signals = sell_signals.copy().astype(int)
    buy_signals = buy_signals.copy().astype(int)

    return sell_signals, buy_signals


def validate(stocks, start_money=1000, start_eq=0, verbose=0):
    """
      Играем по стратегии: при сигнале на покупку покупаем всё, при сигнале на продажу - продаём всё

    Keyword arguments:
      stocks -- Датафрейм с данными о цене (price) и сигналами на покупку (buy_signal) и продажу (sell_signal)
      start_money -- длинное скользящее среднее (Series)
    """

    def _print(txt, priority):
        if priority < verbose:
            print(txt)

    money_hist = list()

    money = start_money
    eq = start_eq
    _print("Денег в начале стратегии: {0}".format(money), 0)

    for row in stocks.iterrows():
        r = row[1]

        if ~np.isnan(r["buy_signal"]):  # сигнал на покупку
            if money > 0:  # деньги есть
                price = r["price"]
                available_eq = np.floor(money / price)
                eq += available_eq
                money -= available_eq * price
                _print(
                    "{0}: Покупаем {1}  по цене {2}, остаток средств: {3}".format(row[0], available_eq, price, money),
                    1)
        if ~np.isnan(r["sell_signal"]):  # сигнал на продажу
            if eq > 0:  # есть что продать
                price = r["price"]
                sell_eq = eq
                eq = 0
                money += sell_eq * price
                _print("{0}: Продаём {1}  по цене {2}, остаток средств: {3}".format(row[0], sell_eq, price, money), 1)
        # записываем историю нашего благосоятояния
        money_hist.append(money)

        # запоминаем последнюю цену
        last_price = r["price"]

    # фиксим прибыль
    money += last_price * eq
    eq = 0

    _print("Денег в конце стратегии: {0}".format(money), 0)

    return money_hist, money, eq


def apply_strategy(data, sell_signals, buy_signals, chosen_sell_signals, chosen_buy_signals):
    """
      Примененяет стратегию

    Keyword arguments:
      data -- Датафрейм с данными о цене price
      sell_signals -- Датафрейм с сигналами на продажу
      buy_signals -- Датафрейм с сигналами на покупку
      chosen_sell_signals -- имя сигнала на продажу
      chosen_buy_signals -- Имя сигнала на покупку
    """
    result = pd.DataFrame()
    result["price"] = data["price"]

    result["sell_signal"] = np.NaN
    result.loc[sell_signals[chosen_sell_signals] > 0, "sell_signal"] = data.loc[
        sell_signals[chosen_sell_signals] > 0, "price"]

    result["buy_signal"] = np.NaN
    result.loc[buy_signals[chosen_buy_signals] > 0, "buy_signal"] = data.loc[
        buy_signals[chosen_buy_signals] > 0, "price"]

    return result
