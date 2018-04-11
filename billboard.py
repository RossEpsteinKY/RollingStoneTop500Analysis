import sqlite3
import re
import requests
from math import pi
from itertools import chain
from collections import namedtuple
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, FuncTickFormatter, FixedTicker, ColumnDataSource
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
from collections import Counter
from operator import itemgetter

#IMPORT THE CSV
def rungraph():
    token = ''
    header = headers = {'Authorization': f'Bearer {token}','Content-type': 'application/json'}


    raw_billboard_csv = requests.get('https://query.data.world/s/ptemuucu25h6vxoaswh5lwg6b6x6oh', 
                                headers=headers)

    raw_billboard_data = raw_billboard_csv.text

    billboard_split = raw_billboard_data.split('\r')
    billboard_split[6]

    album_genre = [r.split(',')[-1] for r in billboard_split[1:]]

    album_year = [r.split(',')[1] for r in billboard_split[1:]]

    album_name = [r.split(',')[2] for r in billboard_split[1:]]

    album_artist = [r.split(',')[3] for r in billboard_split[1:]]

    formatted_albums = list(zip(album_year, album_name, album_artist, album_genre))
    type(formatted_albums[0])
    formatted_albums[0]

    conn = sqlite3.connect('album_info.db')
    cur = conn.cursor() 
    cur.execute('''drop table IF EXISTS album_data''')
    cur.execute('''CREATE TABLE IF NOT EXISTS album_data (album_year TEXT, album_name TEXT, album_artist TEXT, album_genre TEXT)''')

    cur.executemany('INSERT OR REPLACE INTO album_data VALUES (?,?,?,?)', formatted_albums)
    conn.commit()

    most_albums_artist = cur.execute('''SELECT album_artist  FROM 'album_data' GROUP BY album_artist ORDER BY count(album_genre) DESC
    LIMIT 10;''').fetchall()

    most_albums_count = cur.execute(''' SELECT album_artist, count(album_artist) FROM 'album_data' GROUP BY album_artist ORDER BY count(album_genre) DESC LIMIT 15;''').fetchall()

    artist_name, artist_appearances = map(list, zip(*most_albums_count))
    print(artist_name)
    print(artist_appearances)


    x = np.arange(4)



    def millions(x, pos):
        'The two args are the value and tick position'
        return '$%1.1fM' % (x * 1e-6)


    formatter = FuncFormatter(millions)
    y_pos = np.arange(len(artist_name))
    colors = ['r','y','g','b', '#FB00FF','m','#FFA8A8','#9FFFA6',]
    plt.bar(y_pos, artist_appearances, align='center', alpha=0.5, color=colors)
    plt.xticks(y_pos, artist_name, rotation=25)
    plt.ylabel('# of Appearances on Rolling Stone Top 500 Chart')
    plt.title('Top Appearing Artists on Rolling Stone Top 500 Albums of All Time' '\n' 'Name of Artist')
    
    plt.show()

def progstart():
    choice = input('Would you like to run the graph?>  ')
    choice = choice.lower()

    if choice == 'yes':
        print('RUNNING THE GRAPH NOW!')
        rungraph()
    else:
        print("Say 'yes' when ready!")
        progstart()
progstart()
