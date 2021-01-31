import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------------------- Server side logic --------------------------------------
# Get today's date as UTC timestamp
class Asset:
    def __init__(self, ticker):
        self.ticker = ticker
        self.processed_data = self._get_indicators()

    def _get_price_hist(self):
        today = datetime.today().strftime("%d/%m/%Y")
        today = datetime.strptime(today + " +0000", "%d/%m/%Y %z")
        to = int(today.timestamp())
        # Get date ten years ago as UTC timestamp
        #ten_yr_ago = today - relativedelta(years=10)
        one_yr_ago = today - relativedelta(years=1)
        #fro = int(ten_yr_ago.timestamp())
        fro = int(one_yr_ago.timestamp())

        url = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={fro}&period2={to}&interval=1d&events=history".format(
            ticker=self.ticker, fro=fro, to=to)
        data = pd.read_csv(url)

        # Convert date to timestamp and make index
        data.index = data["Date"].apply(lambda x: pd.Timestamp(x))

        #For later use
        return data

    @staticmethod
    def _computeRSI(data, time_window):
        diff = data.diff(1).dropna()  # diff in one field(one day)

        # this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff

        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[diff > 0]

        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[diff < 0]

        # check pandas documentation for ewm
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

        rs = abs(up_chg_avg / down_chg_avg)
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _get_indicators(self):
        data = self._get_price_hist()
        # Get MACD
        # https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
        data['Date'] = data.index
        data['MACD'] = data['Adj Close'].ewm(span=12, adjust=False).mean() - data['Adj Close'].ewm(span=26,
                                                                                                   adjust=False).mean()
        data["MACD_signal"] = data['MACD'].ewm(span=9, adjust=False).mean()
        # data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data['Adj Close'])

        # Get MA10 and MA30
        data["MA_10"] = data["Adj Close"].rolling(window=10).mean()
        data["MA_30"] = data["Adj Close"].rolling(window=30).mean()

        # Get RSI
        # https://tcoil.info/compute-rsi-for-stocks-with-python-relative-strength-index/
        data["RSI"] = self._computeRSI(data['Close'], 14)
        return data

    def OHLC_chart(self):
        fig_data = self.processed_data
        fig = go.Figure(data=[go.Candlestick(x=fig_data['Date'],
                                             open=fig_data['Open'],
                                             high=fig_data['High'],
                                             low=fig_data['Low'],
                                             close=fig_data['Close'])])

        fig.add_trace(
            go.Scatter(
                x=fig_data['Date'],
                y=fig_data['MA_10']
            ))

        fig.add_trace(
            go.Scatter(
                x=fig_data['Date'],
                y=fig_data['MA_30']
            ))

        return fig

    def RSI_chart(self):
        fig_data = self.processed_data
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fig_data['Date'],
                y=fig_data['RSI'], name='RSI'
            ))

        # Standard significance
        fig.add_shape(type="line",
                      x0=min(fig_data['Date']), y0=30, x1=max(fig_data['Date']), y1=30,
                      line=dict(
                          color="LightSeaGreen",
                          width=2,
                          dash="dashdot",
                      )
                      )

        fig.add_shape(type="line",
                      x0=min(fig_data['Date']), y0=70, x1=max(fig_data['Date']), y1=70,
                      line=dict(
                          color="LightSeaGreen",
                          width=2,
                          dash="dashdot",
                      )
                      )

        # Very significant

        fig.add_shape(type="line",
                      x0=min(fig_data['Date']), y0=20, x1=max(fig_data['Date']), y1=20,
                      line=dict(
                          color="red",
                          width=2,
                          dash="dashdot",
                      )
                      )

        fig.add_shape(type="line",
                      x0=min(fig_data['Date']), y0=80, x1=max(fig_data['Date']), y1=80,
                      line=dict(
                          color="red",
                          width=2,
                          dash="dashdot",
                      )
                      )
        return fig

    def MACD_chart(self):
        fig_data = self.processed_data
        #fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_bar(x=fig_data['Date'],
                    y=fig_data['Volume'] / 1000000, name='Volume')

        fig.add_trace(
            go.Scatter(
                x=fig_data['Date'],
                y=fig_data['MACD']
            , name='MACD'), secondary_y=True)

        fig.add_trace(
            go.Scatter(
                x=fig_data['Date'],
                y=fig_data['MACD_signal'], name='MACD Signal Line'
            ), secondary_y=True)

        return fig

# -------------------------------------- Web integration --------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'BTC Dashboard'
app.title = 'BTC Dashboard'

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

asset = Asset('BTC-USD')
fig2 = asset.OHLC_chart()
fig3 = asset.RSI_chart()
fig4 = asset.MACD_chart()

app.layout = html.Div([
    html.Div([html.H1('Technical Analysis dashboard'), html.P('The is the product of a blog post written on Dash. You can read the post and view the source code here.')])
    , html.Div([
        html.Div([
            html.H3('MACD and Volumes'),
            dcc.Graph(id='g1', figure=fig4)
        ], className="six columns"),

        html.Div([
            html.H3('Candlestock OHLC'),
            dcc.Graph(id='g2', figure=fig2)
        ], className="six columns"),
    ], className="row"),
html.Div([
        html.Div([
            html.H3('14-day RSI indication'),
            dcc.Graph(id='g3', figure=fig3)
        ], className="ten columns"),
    ], className="row")
])

if __name__ == '__main__':
    app.run_server(debug=True)
