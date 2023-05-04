import streamlit as st
from db import *
from models import *


def home_page():
    """ Main Page """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<center><img src=https://cdn.pixabay.com/photo/2016/07/26/16/16/wine-1543170_960_720.jpg style='display: block; width: 500px;'></center>", unsafe_allow_html=True)
    with col2:
        st.title("Welcome, we are WinePickers ✨")
        
    st.subheader(f"{st.session_state['profile']['user_name']}, your current location is {st.session_state['profile']['address']}")

    recommendation = recommend_wine(df_embedding, st.session_state['profile']['embeddings'])
    recommendation_df = df_wine.loc[recommendation.iloc[:200].index]

    with st.container():
        st.subheader("Top 5 wines you may like.")
        columns = st.columns(5)
        for i in range(len(columns)):
            url = recommendation_df['url'].values[i]
            wine_name = recommendation_df['wine_name'].values[i]
            columns[i].markdown(f"<center><img src={url} style='display: block; width: 100px;'></center>", unsafe_allow_html=True)
    with st.container():
        columns = st.columns(5)
        for i in range(len(columns)):
            wine_name = recommendation_df['wine_name'].values[i]
            columns[i].markdown(f"<center>{wine_name}</center>", unsafe_allow_html=True)

    with st.container():
        st.subheader("Top wine continents you may like.")
        recommend_continent = best_continent(recommendation_df, model.encoder.embedding)
        columns = st.columns(min(len(recommend_continent), 5))
        for i in range(len(columns)):
            columns[i].markdown(f"{i+1}. {recommend_continent[i]}")
    
    with st.container():
        st.subheader("Top grape breeds you may like.")
        recommend_grapes = best_grapes(recommendation_df, model.encoder.embedding)
        columns = st.columns(min(len(recommend_grapes), 5))
        for i in range(len(columns)):
            columns[i].markdown(f"{i+1}. {recommend_grapes[i]}")
            
    with st.container():
        st.subheader("Top countries you may like.")
        recommend_counties = best_countries(recommendation_df, model.encoder.embedding)
        columns = st.columns(min(len(recommend_counties), 5))
        for i in range(len(columns)):
            columns[i].markdown(f"{i+1}. {recommend_counties[i]}")
    
    # Wine recommendation
    with st.container():
        st.subheader("How about the recommended wine?")
        wine = st.selectbox('', (df_wine['wine_name']), label_visibility='collapsed', key='home page wine select box')
        wine_idx = df_wine[df_wine['wine_name'] == wine].index
        target_wine_embeddings = df_embedding[wine_idx]

        rate = float(st.select_slider('Please select update rate. Higher value will update more.',
                                    options=[f'{0.5*i:.1f}' for i in range(11)], value='2.5', key='home page wine rate'))
        
        update = st.button('Update', key='home page update button')

        if update:
            try:
                updated_embeddings = update_my_vec(st.session_state['profile']['embeddings'], target_wine_embeddings, rate)[0]
                encoded_embeddings = encode_vector(updated_embeddings)
                update_log = update_table('users', {'embeddings': encoded_embeddings}, {'user_name': st.session_state['profile']['user_name']})
                st.session_state['profile']['embeddings'] = updated_embeddings
            except Exception as e:
                st.warning(e, icon="⚠️")


model = load_model()
df_wine, df_embedding = fetch_wines_embeddings()

if 'login_flag' not in st.session_state:
    st.session_state['login_flag'] = 'logout'
if 'profile' not in st.session_state:
    st.session_state['profile'] = ''

if st.session_state['login_flag'] == 'login':
    home_page()
