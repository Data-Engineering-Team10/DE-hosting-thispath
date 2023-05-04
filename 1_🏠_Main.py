import streamlit as st
from db import *
from models import *


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# TODO: 로그인 완성 페이지
def login_page():
    """
    Login Page
    """
    # Create a login form
    st.write("<h1 style='text-align:center'>Login</h1>", unsafe_allow_html=True)
    username = st.text_input("ID", key='login page user username')
    password = st.text_input("Password", type="password")

    # Buttons
    col1, col2 = st.columns([1, 8])
    with col1:
        login_button = st.button("Login", key='login page login button')
    with col2:
        signup_button = st.button("Sign Up", key='login page signup button')

    # Check if the username and password are correct
    if login_button:
        # If correct, change it to main page. Otherwise, warning pops up.
        try:
            query_result = select_table("users", where_dict={'user_name': username, 'password': password})
            
            st.write("<h1 style='text-align:center'>Welcome, {}!</h1>".format(username), unsafe_allow_html=True)
            st.write("<h4 style='text-align:center'>You have successfully logged in.</h4>", unsafe_allow_html=True)
            st.write("<h4 style='text-align:center'>Enjoy your wine!</h4>", unsafe_allow_html=True)
            
            st.session_state['login_flag'] = 'login'
            st.session_state['profile'] = {key: value[0] for key, value in query_result.items()}
        except Exception as e:
            st.warning("Incorrect username or password", icon="⚠️")

    if signup_button:
        st.session_state['login_flag'] = 'signup'

def signup_page():
    """
    Sign up Page
    """
    # Create a sign up form
    st.write("<h1 style='text-align:center'>Sign Up</h1>", unsafe_allow_html=True)
    username = st.text_input("ID", key='signup page username')
    # email = st.text_input("Email")
    # phone = st.text_input("Phone Number")
    password = st.text_input("Password", type="password", key='signup page password')
    address = st.selectbox("Address", ("쌍암동", "오룡동"), key='signup page address')

    # Wine type
    st.markdown("#### What is your favorite wine type?")
    wine_type = st.radio("", ('Red wine', 'White wine'), label_visibility='collapsed', key='signup page wine_type').split()[0]

    # Wine taste
    st.markdown("#### What kind of taste do you prefer?")
    bold = float(st.select_slider('Light(0) ~ Bold(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page bold'))
    tannic = float(st.select_slider('Smooth(0) ~ Tannic(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page tannic'))
    sweet = float(st.select_slider('Dry(0) ~ Sweet(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page sweet'))
    acidic = float(st.select_slider('Soft(0) ~ Acidic(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page acidic'))

    # Buttons
    col1, col2 = st.columns([9, 1])
    with col1:
        signup_button = st.button("Sign Up", key='signup page signup button')
    with col2:
        back_button = st.button("Back", key='signup page back button')

    # Check if the password and confirm password match
    if signup_button:
        try:
            if username == '':
                raise Exception("Please enter ID.")
            if password == '':
                raise Exception("Please enter password.")
            initial_embeddings = get_initial_vec(model, df_wine, wine_type, bold, tannic, sweet, acidic)
            encoded_embeddings = encode_vector(initial_embeddings)
            
            row_dict = {'user_name': username,
                        'password': password,
                        'address': address,
                        'wine_type': wine_type,
                        'bold': bold,
                        'tannic': tannic,
                        'sweet': sweet,
                        'acidic': acidic,
                        'embeddings': encoded_embeddings}
            
            query_result = insert_table("users", row_dict)
            
            st.write("<h1 style='text-align:center'>Welcome, {}!</h1>".format(username), unsafe_allow_html=True)
            st.write("<p style='text-align:center'>You have successfully signed up.</p>", unsafe_allow_html=True)

            st.session_state['login_flag'] = 'login'
            st.session_state['profile'] = {key: value[0] for key, value in query_result.items()}
        except psycopg2.errors.UniqueViolation as e:
            st.warning('ID already exists.', icon="⚠️")
        except Exception as e:
            st.warning(e, icon="⚠️")

    if back_button:
        st.session_state['login_flag'] = 'logout'
        
def my_page():
    """
    My Page
    """
    st.write("<h1 style='text-align:center'>Welcome, {}!</h1>".format(st.session_state['profile']['user_name']), unsafe_allow_html=True)
    # TODO: 와인 정보 페이지
    st.markdown("# About wines...")



    st.markdown("### Do you want to reset your wine preference?")
    with st.expander("See more details."):
    # Wine type
        st.markdown("#### What is your favorite wine type?")
        wine_type = st.radio("", ('Red wine', 'White wine'), label_visibility='collapsed', key='signup page wine_type').split()[0]

        # Wine taste
        st.markdown("#### What kind of taste do you prefer?")
        bold = float(st.select_slider('Light(0) ~ Bold(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page bold'))
        tannic = float(st.select_slider('Smooth(0) ~ Tannic(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page tannic'))
        sweet = float(st.select_slider('Dry(0) ~ Sweet(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page sweet'))
        acidic = float(st.select_slider('Soft(0) ~ Acidic(1)', options=[f'{0.1*i:.1f}' for i in range(11)], value='0.5', key='signup page acidic'))

        update = st.button('Update', key='home page update button')

        if update:
            try:
                initial_embeddings = get_initial_vec(model, df_wine, wine_type, bold, tannic, sweet, acidic)
                encoded_embeddings = encode_vector(initial_embeddings)
                update_log = update_table('users', {'embeddings': encoded_embeddings}, {'user_name': st.session_state['profile']['user_name']})
                st.session_state['profile']['embeddings'] = initial_embeddings
                st.markdown('Update Done!')
            except Exception as e:
                st.warning(e, icon="⚠️")
        

model = load_model()
df_wine, df_embedding = fetch_wines_embeddings()

if 'login_flag' not in st.session_state:
    st.session_state['login_flag'] = 'logout'
if 'profile' not in st.session_state:
    st.session_state['profile'] = ''

if st.session_state['login_flag'] == 'logout':
    login_page()
elif st.session_state['login_flag'] == 'signup':
    signup_page()
else:
    my_page()
