# imports
import streamlit as st
import pandas as pd
import numpy as np

# %% STREAMLIT
# Set configuration
st.set_page_config(page_title="PCB Inspector",
                   page_icon="🔍",
                   initial_sidebar_state="expanded",
                   # layout="wide"
                   )

# set colors: These has to be set on the setting menu online
    # primary color: #FF4B4B, background color:#0E1117
    # text color: #FAFAFA, secindary background color: #E50914

# Set the logo of app
# st.sidebar.image("pcb_inspector_logo.png", width=300, clamp=True)
# welcome_img = Image.open('welcome_page_img.png')
# st.image(welcome_img)
st.markdown("""
# 🔍 PCB Inspector 🔍
""")

# %% APP WORKFLOW
st.markdown("""
### How may we help you?
"""
)

# file uploader
uploaded_file = st.file_uploader("Choose a file")

# Popularity based recommender system
genre_default, n_default = None, None
pop_based_rec = st.checkbox("Show me the all time favourites",
                            False,
                            help="Movies that are liked by many people")


if pop_based_rec:
    st.markdown("### Select Genre and Nr of movie recommendations")
    genre_default, n_default = None, 5
    with st.form(key="pop_form"):
        # genre_default, year_default = ['Any'], ['Any']
        genre = st.multiselect(
                "Genre",
                options=genre_list,
                help="Select the genre of the movie you would like to watch",
                default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        key="n",
                        help="How many movie recommendations would you like to receive?",
                        )

        submit_button_pop = st.form_submit_button(label="Submit")


    if submit_button_pop:
        popular_movie_recs = popular_n_movies(nr_rec, genre[0])
        st.table(popular_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

item_based_rec = st.checkbox("Show me a movie like this",
                             False,
                             help="Input some movies and we will show you similar ones")
from random import choice
short_movie_list = ['Prestige, The (2006)', 'Toy Story (1995)',
                    'No Country for Old Men (2007)']
if item_based_rec:
    st.markdown("### Tell us a movie you like:")
    with st.form(key="movie_form"):
        movie_name = st.multiselect(label="Movie name",
                                    # options=movie_list,
                                    options=pd.Series(movie_list),
                                    help="Select a movie you like",
                                    key='item_select',
                                    default=choice(short_movie_list))

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec_movie",
                           help="How many movie recommendations would you like to receive?",
                           )

        submit_button_movie = st.form_submit_button(label="Submit")

    if submit_button_movie:
        st.write('Because you like {}:'.format(movie_name[0]))

        item_movie_recs = item_n_movies(movie_name[0], nr_rec)
        st.table(item_movie_recs)

# to put some space in between options
st.write("")
st.write("")
st.write("")

user_based_rec = st.checkbox("I want to get personalized recommendations",
                             False,
                             help="Login to get personalized recommendations")

if user_based_rec:
    st.markdown("### Please login to get customized recommendations just for you")
    genre_default, n_default = None, 5
    with st.form(key="user_form"):

        user_id = st.number_input("Please enter your user id", step=1,
                                  min_value=1)
        # genre_default, year_default = ['Any'], ['Any']
        # genre = st.multiselect(
        #         "Genre",
        #         options=genre_list,
        #         help="Select the genre of the movie you would like to watch",
        #         default=genre_default)

        nr_rec = st.slider("Number of recommendations",
                           min_value=1,
                           max_value=20,
                           value=5,
                           step=1,
                           key="nr_rec",
                           help="How many movie recommendations would you like to receive?",
                           )

        submit_button_user = st.form_submit_button(label="Submit")


    if submit_button_user:
        user_movie_recs = user_n_movies(user_id, nr_rec)
        st.table(user_movie_recs)
