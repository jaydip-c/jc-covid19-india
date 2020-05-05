import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

import os
__path__=[os.path.dirname(os.path.abspath(__file__))]
from covid import Covid

url1 = "https://www.worldometers.info/coronavirus/#countries"
url2 = "https://pomber.github.io/covid19/timeseries.json"

c = Covid()
wdf = c.get_world_data(url1)
idf = c.get_india_data(url1, url2)
adjusted_dates, icases, future_forcast, linear_pred, df1 = c.predict_india(url1, url2)

WCTotal = wdf.at[wdf.index[-1], 'TotalCases']
WDTotal = wdf.at[wdf.index[-1], 'TotalDeaths']
WRTotal = wdf.at[wdf.index[-1], 'TotalRecovered']
WATotal = wdf.at[wdf.index[-1], 'ActiveCases']

new_df = wdf[:-1]
TCountry = new_df['Country']
ACases = new_df['ActiveCases']
TotCasesDeaths = new_df['TotalDeaths']
TRecovered = new_df['TotalRecovered']

ICTotal = idf.at[idf.index[-2], 'confirmed']
dICTotal = idf.at[idf.index[-3], 'confirmed']
cDelta = ICTotal - dICTotal
if cDelta >= 0:
    sc = f"Increased by {cDelta}"
else:
    sc = f"Decreased by {cDelta}"

IDTotal = idf.at[idf.index[-2], 'deaths']
dIDTotal = idf.at[idf.index[-3], 'deaths']
dDelta = IDTotal - dIDTotal
if dDelta >= 0:
    sd = f"Increased by {dDelta}"
else:
    sd = f"Decreased by {dDelta}"

IRTotal = idf.at[idf.index[-2], 'recovered']
dIRTotal = idf.at[idf.index[-3], 'recovered']
rDelta = IRTotal - dIRTotal
if rDelta >= 0:
    sr = f"Increased by {rDelta}"
else:
    sr = f"Decreased by {rDelta}"

IATotal = idf.at[idf.index[-2], 'active']
dIATotal = idf.at[idf.index[-3], 'active']
aDelta = IATotal - dIATotal
if aDelta >= 0:
    sa = f"Increased by {aDelta}"
else:
    sa = f"Decreased by {aDelta}"

Idate = pd.to_datetime(idf['date'])
Iconfirmed = idf['confirmed']
Iactive = idf['active']
Ideaths = idf['deaths']
Irecovered = idf['recovered']

a = pd.DataFrame(adjusted_dates, columns = ['adjusted_dates'])
b = pd.DataFrame(icases, columns = ['icases'])
c = pd.DataFrame(future_forcast, columns = ['future_forcast'])
d = pd.DataFrame(linear_pred, columns = ['linear_pred'])
df2 = pd.concat([a, b, c, d], axis=1)
df = df2[["adjusted_dates", "icases", "future_forcast", "linear_pred"]]
a = df["adjusted_dates"]
b = df["icases"]
c = df["future_forcast"]
d = df["linear_pred"]

cell1 = dcc.Graph(
            id='wbar',
            figure={
                'data':[
                    {'x':TCountry, 'y':ACases, 'type':'bar', 'name':'Active'},
                    {'x':TCountry, 'y':TotCasesDeaths, 'type':'bar', 'name':'Deaths'},
                    {'x':TCountry, 'y':TRecovered, 'type':'bar', 'name':'Recovered'}
                ],
                'layout':{
                    'title':'World Top 10 & India',
                    'barmode':'stack',
                    'height':400,
                }
            }
        )

cell2 = dash_table.DataTable(
            id='wtable',
            columns=[{"name":i, "id":i} for i in new_df.columns],
            data=new_df.to_dict('rows'),
            style_table={
                'height': '400px',
                'overflowY': 'auto'
            },
            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            }
        )

cell3 = dcc.Graph(
            id='iline',
            figure={
                'data':[
                    {'x':Idate, 'y':Iactive, 'type':'lines', 'name':'Active'},
                    {'x':Idate, 'y':Ideaths, 'type':'lines', 'name':'Deaths'},
                    {'x':Idate, 'y':Irecovered, 'type':'lines', 'name':'Recovered'}
                ],
                'layout':{
                    'title':'India Total Active, Recovered and Death Cases',
                    'height':400,
                }
            }
        )

cell4 = dash_table.DataTable(
            id='itable',
            columns=[{"name":i, "id":i} for i in idf.columns],
            data=idf.to_dict('rows'),
            style_table={
                'height': '400px',
                'overflowY': 'auto'
            },
            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            }
        )

cell5 = dcc.Graph(
            id='pline',
            figure={
                'data':[
                    {'x':a, 'y':b, 'type':'lines', 'name':'Real cases'},
                    {'x':c, 'y':d, 'type':'lines', 'name':'Polynomial Regression Predictions'}
                ],
                'layout':{
                    'title':'Cases in India: Predicting Next 5 days',
                    'height':400,
                }
            }
        )

cell6 = dash_table.DataTable(
            id='ptable',
            columns=[{"name":i, "id":i} for i in df1.columns],
            data=df1.to_dict('rows'),
            style_table={
                'height': '800px',
                'overflowY': 'auto'
            },
            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'font_size': 36,
            }
        )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

dash_colors = {
    'background': '#111111',
    'text': '#BEBEBE',
    'grid': '#333333',
    'red': '#BF0000',
    'blue': '#466fc2',
    'green': '#5bc246',
    'yellow': '#f0f00c',
}

app.title = 'COVID-19 India'

app.layout = html.Div(
    style={
        'backgroundColor': dash_colors['background']
    },
    children=[
        html.H1(
            children='Covid-19 World Data Analysis',
            style={
                'textAlign':'center',
                'color': dash_colors['text']
            }
        ),

        html.Div(html.H3(['Worldwide Total Cases:', html.Br(), WCTotal],
                    style={
                    'textAlign':'center',
                    'color': dash_colors['blue'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'25%',
                    'float': 'left',
                    'backgroundColor': dash_colors['background'],
                })
        ),

        html.Div(html.H3(['Worldwide Active Cases:', html.Br(), WATotal],
                    style={
                    'textAlign':'center',
                    'color': dash_colors['yellow'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'25%',
                    'float': 'left',
                    'backgroundColor': dash_colors['background'],
                })
        ),

        html.Div(html.H3(['Worldwide Total Death:', html.Br(), WDTotal],
                    style={
                    'textAlign':'center',
                    'color': dash_colors['red'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'22%',
                    'float': 'left',
                    'backgroundColor': dash_colors['background'],
                })
        ),

        html.Div(html.H3(['Worldwide Total Recovered:', html.Br(), WRTotal],
                    style={
                    'textAlign':'center',
                    'color': dash_colors['green'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'25%',
                    'float': 'left',
                    'backgroundColor': dash_colors['background'],
                })
        ),

        html.Div(children=cell1,
                    style={
                    'backgroundColor': dash_colors['background'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'48%',
                    'float': 'left',
                }
        ),

        html.Div(children=cell2,
                    style={
                    'backgroundColor': dash_colors['background'],
                    'border-style':'solid',
                    'display':'inline-block',
                    'width':'50%',
                    'float': 'left',
                }
        ),

        html.Div(
            style={
                'backgroundColor': dash_colors['background']
            },
            children=[
                html.H1(
                    children='Covid-19 India Data Analysis',
                    style={
                        'textAlign':'center',
                        'color': dash_colors['text']
                    }
                ),

                html.Div(html.H3(['India Total Cases:', html.Br(), ICTotal, html.Br(), sc],
                            style={
                            'textAlign':'center',
                            'color': dash_colors['blue'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'25%',
                            'float': 'left',
                            'backgroundColor': dash_colors['background'],
                        })
                ),

                html.Div(html.H3(['India Active Cases:', html.Br(), IATotal, html.Br(), sa],
                            style={
                            'textAlign':'center',
                            'color': dash_colors['yellow'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'25%',
                            'float': 'left',
                            'backgroundColor': dash_colors['background'],
                        })
                ),

                html.Div(html.H3(['India Total Death:', html.Br(), IDTotal, html.Br(), sd],
                            style={
                            'textAlign':'center',
                            'color': dash_colors['red'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'22%',
                            'float': 'left',
                            'backgroundColor': dash_colors['background'],
                        })
                ),

                html.Div(html.H3(['India Total Recovered:', html.Br(), IRTotal, html.Br(), sr],
                            style={
                            'textAlign':'center',
                            'color': dash_colors['green'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'25%',
                            'float': 'left',
                            'backgroundColor': dash_colors['background'],
                        })
                ),

                html.Div(children=cell3,
                            style={
                            'backgroundColor': dash_colors['background'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'48%',
                            'float': 'left',
                        }
                ),

                html.Div(children=cell4,
                            style={
                            'backgroundColor': dash_colors['background'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'50%',
                            'float': 'left',
                        }
                ),
            ]
        ),

        html.Div(
            style={
                'backgroundColor': dash_colors['background']
            },

            children=[
                html.H1(
                    children='Covid-19 India Data Prediction',
                    style={
                        'textAlign':'center',
                        'color': dash_colors['text']
                    }
                ),

                html.Div(children=cell5,
                            style={
                            'backgroundColor': dash_colors['background'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'48%',
                            'float': 'left',
                        }
                ),

                html.Div(children=cell6,
                            style={
                            'backgroundColor': dash_colors['background'],
                            'border-style':'solid',
                            'display':'inline-block',
                            'width':'50%',
                            'float': 'left',
                            'height':'400px'
                        }
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
