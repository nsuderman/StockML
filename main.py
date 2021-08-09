import streamlit as st
from streamlit.server.server import Server
from streamlit.report_thread import get_report_ctx
from finviz.screener import Screener
import pandas as pd
import ta
from yahooquery import Ticker
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
import os
from ml import ml
from plotly.subplots import make_subplots
import numpy as np


def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def watchers():
    # get report context
    _max_width_()
    ctx = get_report_ctx()
    # get session id
    session_id = ctx.session_id

    # get session
    server = Server.get_current()
    session_info = server._session_info_by_id.get(session_id)
    session = session_info.session

    # register watcher
    session._local_sources_watcher._register_watcher(
        os.path.join(os.path.dirname(__file__), './ml/ml.py'),'dummy:ml/ml.py')
    # session._local_sources_watcher._register_watcher(
    #     os.path.join(os.path.dirname(__file__), './modules/segmentation.py'), 'dummy:modules/segmentation.py')
    # session._local_sources_watcher._register_watcher(
    #     os.path.join(os.path.dirname(__file__), './modules/customer_lifetime_value.py'), 'dummy:modules/customer_lifetime_value.py')


def get_ticker_data(ticker):
    momentum = 45

    t_df = Ticker(ticker).history(start=datetime.now() + timedelta(-365),
                                            end=datetime.now(),
                                            interval='1d',
                                            adj_ohlc=True)
    t_df['ticker'] = t_df._get_label_or_level_values('symbol')
    t_df.index = pd.to_datetime(t_df._get_label_or_level_values('date'))

    ticker = t_df.copy()

    t_df['momentum'] = t_df['close'].rolling(45).apply(ml.momentum)
    t_df['momentum2'] = t_df['close'].rolling(45).apply(ml.momentum2)

    bband = ta.volatility.BollingerBands(t_df['close'],window=20,window_dev=2)
    t_df['bol_mid'] = bband.bollinger_mavg()
    t_df['bol_low'] = bband.bollinger_lband()
    t_df['bol_high'] = bband.bollinger_hband()

    adx = ta.trend.ADXIndicator(t_df['high'],t_df['low'],t_df['close'],window=16)
    t_df['adx'] = adx.adx()
    t_df['adx_neg'] = adx.adx_neg()
    t_df['adx_pos'] = adx.adx_pos()

    t_df['ema_slow'] = ta.trend.EMAIndicator(t_df['close'],window=16).ema_indicator()
    t_df['ema_fast'] = ta.trend.EMAIndicator(t_df['close'], window=9).ema_indicator()

    macd = ta.trend.MACD(t_df['close'],12,26,9)
    t_df['macd'] = macd.macd()
    t_df['macd_hist'] = macd.macd_diff()
    t_df['macd_signal'] = macd.macd_signal()

    t_df['rsi'] = ta.momentum.StochRSIIndicator(t_df['close']).stochrsi()

    t_df['signal'] = 'Hold'
    t_df.loc[(t_df['adx'] > 15) & (t_df['momentum'] > 1),'signal'] = 'Buy'
    t_df.loc[(t_df['adx'] < 15) | (t_df['momentum'] < 1), 'signal'] = 'Sell'
    #t_df['Signal'] = np.where((t_df['adx'] < 15) | (t_df['momentum2'] < 1), 'Sell')


    return t_df


def run_st():
    st.set_page_config(page_title="Stock", layout='wide', initial_sidebar_state='auto')
    # watchers()

    st.sidebar.header('Navigation')
    st.sidebar.write('')  # Line break

    side_menu_selectbox = st.sidebar.selectbox('Menu', ('Home',
                                                        'Filter',
                                                        'Analysis')
                                               )
    st.sidebar.write('___')

    if side_menu_selectbox == 'Home':
        st.header("Project Overview")
        st.write('')
        st.write('This project was put together to allow for quick stock analysis.')
        st.write('')

    elif side_menu_selectbox == 'Filter':
        st.header('Stock Filter')
        st.write('')

        price = ('sh_price_u1'
                 , 'sh_price_u2'
                 , 'sh_price_u3'
                 , 'sh_price_u4'
                 , 'sh_price_u5'
                 , 'sh_price_u7'
                 , 'sh_price_u10'
                 , 'sh_price_u15'
                 , 'sh_price_u20'
                 , 'sh_price_u30'
                 , 'sh_price_u40'
                 , 'sh_price_u50'
                 , 'sh_price_o1'
                 , 'sh_price_o2'
                 , 'sh_price_o3'
                 , 'sh_price_o4'
                 , 'sh_price_o5'
                 , 'sh_price_o7'
                 , 'sh_price_o10'
                 , 'sh_price_o15'
                 , 'sh_price_o20'
                 , 'sh_price_o30'
                 , 'sh_price_o40'
                 , 'sh_price_o50'
                 , 'sh_price_o60'
                 , 'sh_price_o70'
                 , 'sh_price_o80'
                 , 'sh_price_o90'
                 , 'sh_price_o100'
                 , 'sh_price_1to5'
                 , 'sh_price_1to20'
                 , 'sh_price_5to10'
                 , 'sh_price_5to20'
                 , 'sh_price_5to50'
                 , 'sh_price_10to20'
                 , 'sh_price_10to50'
                 , 'sh_price_20to50'
                 , 'sh_price_50to100')
        volume = ('sh_curvol_u50'
                  , 'sh_curvol_u100'
                  , 'sh_curvol_u500'
                  , 'sh_curvol_u750'
                  , 'sh_curvol_u1000'
                  , 'sh_curvol_o50'
                  , 'sh_curvol_o100'
                  , 'sh_curvol_o500'
                  , 'sh_curvol_o1000'
                  , 'sh_curvol_o2000'
                  , 'sh_curvol_o5000'
                  , 'sh_curvol_o10000'
                  , 'sh_curvol_o20000')

        price_selectbox = st.sidebar.selectbox('Price', price, index=price.index('sh_price_5to20'))
        volume_selectbox = st.sidebar.selectbox('Volume', volume, index=volume.index('sh_curvol_o2000'))
        if (st.sidebar.button("Get New Filtered Stocks")):
            with st.spinner('Fetching New Stock List'):
                filters = [price_selectbox, volume_selectbox]
                stock_list = Screener(filters=filters, table='Overview', order='-volume', )
                stock_list.to_csv('stocks.csv')

        df = pd.read_csv('stocks.csv')
        df['Price'] = df['Price'].astype(float).fillna(0.0)
        df['Volume_int'] = df['Volume'].str.replace(",", "").astype(int).fillna(0.0)
        st.subheader(f'{len(df)} Rows Returned')
        st.dataframe(df)

    elif side_menu_selectbox == 'Analysis':
        st.header('Stock Analysis')
        st.write('')

        df = pd.read_csv('stocks.csv')
        stocks = df['Ticker']
        stock_select_box = st.sidebar.selectbox('Stock', stocks)
        if (st.sidebar.button("Get Stock Data")):
            with st.spinner('Fetching Stock Price'):
                t_df = get_ticker_data(stock_select_box)
            t_df = t_df[90:]

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=(t_df['ticker'].iloc[0],
                                                                                   'Volume','MACD','RSI'),
                                vertical_spacing=0.1, row_width=[0.5, 0.5, 0.2, 1])

            fig.add_trace(go.Scatter(x=t_df.index,
                                     y=t_df['close'],
                                     name='Close'),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=t_df.index,
                                     y=t_df['bol_high'],
                                     line_color='gray',
                                     line={'dash': 'dash'},
                                     name='upper band',
                                     opacity=0.2),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=t_df.index,
                                     y=t_df['bol_low'],
                                     line_color='gray',
                                     line={'dash': 'dash'},
                                     fill='tonexty',
                                     fillcolor='rgba(192,192,192,0.2)',
                                     name='lower band',
                                     opacity=0.2),
                          row=1, col=1)

            fig.add_trace(go.Bar(x=t_df.index, y=t_df['volume'], showlegend=False),
                          row=2, col=1)

            fig.add_trace(go.Bar(x=t_df.index, y=t_df['macd_hist'],name='MACD Hist'), row=3, col=1)
            fig.add_trace(go.Scatter(x=t_df.index, y=t_df['macd'], name='MACD'), row=3, col=1)
            fig.add_trace(go.Scatter(x=t_df.index, y=t_df['macd_signal'], name='MACD Signal'), row=3, col=1)

            fig.add_trace(go.Scatter(x=t_df.index, y=t_df['rsi'], name='RSI_Stoc'), row=4, col=1)
            fig.add_hline(y=0.8,line_color='red',line_width=1,row=4, col=1)
            fig.add_hline(y=0.2, line_color='red',line_width=1, row=4, col=1)

            fig.update_layout(
                # title=t_df['ticker'].iloc[0],
                # title_x=0.5,
                height=900,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.write(t_df.iloc[::-1])


#-m streamlit.cli run --
if __name__ == '__main__':
    run_st()






