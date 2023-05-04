import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium, folium_static
import random
import math
import numpy as np
from models import *
from db import *
#import toy_markets

# TODO : TOY MARKET DB 에서 가져오기, 지금은 임시로 대충 해놓음

#wine_info = pd.read_csv('./.streamlit/vivino_dataset.csv')
toy_data_info = pd.read_csv('./pages/Markets.csv')
wine_info = pd.read_csv('./pages/wines.csv')

model = load_model()
df_user, df_wine, df_embedding = fetch_data()

#toy_data_info = toy_markets.toy_df
#toy_data_info = pd.DataFrame(toy_data_info)



# sinae : df_wine -> wine_info 로 변경
def main_page():

    col1, col2 = st.columns([4,3])

    with col1:
        st.title('Which wine are you looking for?🍷')
        wine = st.selectbox('', (wine_info['wine_name']), label_visibility='collapsed')
        #wine_idx = toy_data_info[toy_data_info['name'] == wine].index

        if st.button('Search'):
            if wine.strip():  # 코드 입력란에 공백이 아닌 문자가 입력되었을 경우
                with col2:
                    search_page(wine)
            else:
                st.error('Please enter a valid search term.')  # 코드 입력란에 공백이 입력되었을 경우 에러 메시지 출력


    st.title('지도 🎈')
    m = folium.Map(location=[35.228956, 126.843181], zoom_start=16)
    folium.Marker(
        [35.228956, 126.843181],
        popup='GIST',
        tooltip='Dasan'
    ).add_to(m)
    
    inventoryA=''
    for i in range(18) : # 18개
        inventoryA = inventoryA + toy_data_info['name'][i] + '\n - '

    markerA = folium.Marker(
        [35.22115148181801, 126.84508234413954],
        popup=folium.Popup("INVENTORY 💖 " + inventoryA[:-2], max_width=250),
        tooltip='🌷 MARKET A'
    ).add_to(m)

    inventoryB=''
    for i in range(18, 48) : # 30개
        inventoryB = inventoryB + toy_data_info['name'][i] + '\n - '

    markerB = folium.Marker(
        [35.22359306367261, 126.85141562924461],
        popup=folium.Popup("INVENTORY 💖 " + inventoryB[:-2], max_width=250),
        tooltip='🌼 MARKET B'
    ).add_to(m)

    inventoryC=''
    for i in range(48, 63) : #14개
        inventoryC = inventoryC + toy_data_info['name'][i] + '\n - '

    markerC = folium.Marker(
        [35.221234713907336, 126.8540341090701],
        popup=folium.Popup("INVENTORY 💖 " + inventoryC[:-2], max_width=300),
        tooltip='🌻 MARKET C'
    ).add_to(m)

    folium.Marker(
        location=[35.234738, 126.838680],
        icon=folium.Icon(color="red"),
        popup=folium.Popup("I'm a red marker", max_width=300),
        tooltip='Red Marker'
    ).add_to(m)

    st_data = st_folium(m, width=725)

    st.title('💗 와인 매장을 보여드릴게요 🐾')
    st.markdown(" ")
    st.markdown(" ")
            
    if st.button('🌷 MARKET A'):
        st.session_state['main_page'] = 'page1'
        st.dataframe(toy_data_info.loc[:17, ['name', 'cost', 'min_cost', 'max_cost']])

            
    if st.button('🌼 MARKET B'):
        st.session_state['main_page'] = 'page2'
        st.dataframe(toy_data_info.loc[18:47, ['name', 'cost', 'min_cost', 'max_cost']])


        
    if st.button('🌻 MARKET C'):
        st.session_state['main_page'] = 'page3'
        st.dataframe(toy_data_info.loc[48:63, ['name', 'cost', 'min_cost', 'max_cost']])

mins = 95.1
maxs = 99.1
def search_page(code):

    # 와인 재고 toy data
    inventoryA = toy_data_info.loc[:17, ['name', 'cost']]
    inventoryB = toy_data_info.loc[18:38, ['name', 'cost']]
    inventoryC = toy_data_info.loc[38:63, ['name', 'cost']]
    
    if code in str(inventoryA.values):
        st.write("**<span style='background-color: #A9D0F5;'>MAERKET A </span>** 에 찾는 와인이 있어용 ~", unsafe_allow_html=True)
        
        result = inventoryA[inventoryA['name'].str.contains(code)]

        if not result.empty:
            st.write(f"**NAME**: {result.iloc[0]['name']}")
            cost = result.iloc[0]['cost']
            min_cost = float(cost) * mins/100
            min_cost = math.trunc(min_cost)
            min_cost = '{:,}'.format(min_cost)

            max_cost = float(cost) * maxs/100
            max_cost = math.trunc(max_cost)
            max_cost = '{:,}'.format(max_cost)

            cost = '{:,}'.format(cost)
            st.write('**✨BEST PRICE✨ : ₩**', cost)
            st.write('**Online Price Min/Max : ₩**', min_cost, '|', max_cost)
            st.write('---------------------')
        else:
            st.write(f"**No results found for \"{code}\"**")

    if code in str(inventoryB.values):
        st.write("**<span style='background-color: #81BEF7;'>MAERKET B </span>** 에 찾는 와인이 있어용 ~", unsafe_allow_html=True)
        result = inventoryB[inventoryB['name'].str.contains(code)]
        if not result.empty:
            st.write(f"**NAME**: {result.iloc[0]['name']}")
            cost = result.iloc[0]['cost']
            min_cost = float(cost) * mins/100
            min_cost = math.trunc(min_cost)
            min_cost = '{:,}'.format(min_cost)

            max_cost = float(cost) * maxs/100
            max_cost = math.trunc(max_cost)
            max_cost = '{:,}'.format(max_cost)

            cost = '{:,}'.format(cost)
            st.write('**✨BEST PRICE✨ : ₩**', cost)
            st.write('**Online Price Min/Max : ₩**', min_cost, '|', max_cost)
            st.write('---------------------')
        else:
            st.write(f"**No results found for \"{code}\"**")

    if code in str(inventoryC.values):
        st.write("**<span style='background-color: #E3CEF6;'>MAERKET C </span>** 에 찾는 와인이 있어용 ~", unsafe_allow_html=True)
        result = inventoryC[inventoryC['name'].str.contains(code)]
        if not result.empty:
            st.write(f"**NAME**: {result.iloc[0]['name']}")
            cost = result.iloc[0]['cost']
            min_cost = float(cost) * mins/100
            min_cost = math.trunc(min_cost)
            min_cost = '{:,}'.format(min_cost)

            max_cost = float(cost) * maxs/100
            max_cost = math.trunc(max_cost)
            max_cost = '{:,}'.format(max_cost)

            cost = '{:,}'.format(cost)
            st.write('**✨BEST PRICE✨ : ₩**', cost)
            st.write('**Online Price Min/Max : ₩**', min_cost, '|', max_cost)
        else:
            st.write(f"**No results found for \"{code}\"**")


    elif code not in str(inventoryA.values) and code not in str(inventoryB.values) and code not in str(inventoryC.values):
        st.write(f"**No results found for \"{code}\"**")


def page1():

    col1, col2, col3 = st.columns(3)

    n=6

    with col1:

        for i in range(n):
            
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)

        
    with col2:

        for i in range(n, 2*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)



    with col3:

        for i in range(2*n, 3*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)

    if st.button('뒤로가기'):
        st.session_state['main_page'] = 'main_page'


def page2():
    
    col1, col2, col3 = st.columns(3)

    n=10

    with col1:

        for i in range(18 , 18 + 1*n):
            
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)

        
    with col2:

        for i in range(18 + 1*n, 18 + 2*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)



    with col3:

        for i in range(18 + 2*n, 18 + 3*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)


    if st.button('뒤로가기'):
        st.session_state['main_page'] = 'main_page'


def page3():
    col1, col2, col3 = st.columns(3)

    n=4

    with col1:

        for i in range(48 , 48 + 1*n):
            
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)


        
    with col2:

        for i in range(48 + 1*n, 48 + 2*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)



    with col3:

        for i in range(48 + 2*n, 48 + 3*n):
            st.write(st.session_state.get('message', ''))

            st.image(toy_data_info['imgurl'][i], caption=toy_data_info['name'][i], width = 100)
            st.write('**type** : ', toy_data_info['type'][i])
            st.write('**city** : ', toy_data_info['city'][i])
            date = toy_data_info['date'][i]
            st.write('**date** :', date)
            
            cost = toy_data_info['cost'][i]
            min_cost = toy_data_info['min_cost'][i]
            max_cost = toy_data_info['max_cost'][i]

            cost = '{:,}'.format(toy_data_info['cost'][i])
            min_cost = '{:,}'.format(toy_data_info['min_cost'][i])
            max_cost = '{:,}'.format(toy_data_info['max_cost'][i])

            st.write('**cost : ₩**', cost)
            st.write('**min/max : ₩**', min_cost, '|', max_cost)


    if st.button('뒤로가기'):
        st.session_state['main_page'] = 'main_page'


st.session_state.setdefault('main_page', 'main_page')

if st.session_state['main_page'] == 'main_page':
    main_page()
elif st.session_state['main_page'] == 'page1':
    page1()
elif st.session_state['main_page'] == 'page2':
    page2()
elif st.session_state['main_page'] == 'page3':
    page3()