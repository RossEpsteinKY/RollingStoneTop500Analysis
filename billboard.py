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


def rungraph():
    token = ''
    header = headers = {'Authorization': f'Bearer {token}','Content-type': 'application/json'}

    #IMPORT THE CSV
    raw_billboard_csv = requests.get('https://query.data.world/s/ptemuucu25h6vxoaswh5lwg6b6x6oh', 
                                headers=headers)

    raw_billboard_data = raw_billboard_csv.text

    #Split Data
    billboard_split = raw_billboard_data.split('\r')
    billboard_split[6]

    album_genre = [r.split(',')[-1] for r in billboard_split[1:]]

    album_year = [r.split(',')[1] for r in billboard_split[1:]]

    album_name = [r.split(',')[2] for r in billboard_split[1:]]

    album_artist = [r.split(',')[3] for r in billboard_split[1:]]

    #Zip data in preperation for insertion into database
    formatted_albums = list(zip(album_year, album_name, album_artist, album_genre))
    type(formatted_albums[0])
    formatted_albums[0]

    #Create SQLITE database
    conn = sqlite3.connect('album_info.db')
    cur = conn.cursor() 
    cur.execute('''drop table IF EXISTS album_data''')
    cur.execute('''CREATE TABLE IF NOT EXISTS album_data (album_year TEXT, album_name TEXT, album_artist TEXT, album_genre TEXT)''')

    #Insert info into SQLITE database
    cur.executemany('INSERT OR REPLACE INTO album_data VALUES (?,?,?,?)', formatted_albums)
    conn.commit()

    #Select and count data from database
    most_albums_count = cur.execute(''' SELECT album_artist, count(album_artist) FROM 'album_data' GROUP BY album_artist ORDER BY count(album_genre) DESC LIMIT 15;''').fetchall()

    #Split data into separate lists
    artist_name, artist_appearances = map(list, zip(*most_albums_count))
    print(artist_name)
    print(artist_appearances)

    #Setup bar graph
    x = np.arange(4)
    y_pos = np.arange(len(artist_name))
    colors = ['r','y','g','b', '#FB00FF','m','#FFA8A8','#9FFFA6',]
    plt.bar(y_pos, artist_appearances, align='center', alpha=0.5, color=colors)
    plt.xticks(y_pos, artist_name, rotation=25)
    plt.ylabel('# of Appearances on Rolling Stone Top 500 Chart')
    plt.title('Top Appearing Artists on Rolling Stone Top 500 Albums of All Time' '\n' 'Name of Artist')
    
    #Show Graph
    plt.show()
#Program created with a yes/no run prompt to prevent any chance of an infinite loop
def progstart():
    choice = input('Would you like to run the graph?>  ')
    choice = choice.lower()

    if choice == 'yes':
        print('RUNNING THE GRAPH NOW!')
        rungraph()
    else:
        print("Say 'yes' when ready!")
        progstart()

#Begin program running
progstart()
