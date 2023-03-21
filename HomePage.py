import streamlit as st
import filepaths
from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page


####################################################### INITIALIZATION ###############################################################
# init the styles of fonts
homepage = '<p style="font-family:Arial Black; color:#262730; text-shadow: 1px 1px 2px Snow; font-size: 250%;"><strong>Homepage 🏠</strong></p>'
comp = '<p style="font-family:Arial Black; color:#262730; text-shadow: 1px 1px 2px Snow; font-size: 250%;"><strong>Chose competition🏆</strong></p>'
title = '<p style="font-family:Arial Black; color:Chocolate; text-shadow: 6px 6px 10px Black; font-size: 320%; text-align: center;">SPORTS HIGHLIGHT GENERATOR 🏀</p>'

# Initialization of the timetag variable
if "timetags" not in st.session_state:
    st.session_state['timetags'] = []
# Initialization of the fps variable
if "fps" not in st.session_state:
    st.session_state['fps'] = 25
# Initialization of the flag variable
if "flag" not in st.session_state:
    st.session_state['flag'] = False
# Initialization of the event variable
if "the_event" not in st.session_state:
    st.session_state['the_event'] = "0"
# Initialization of the game variable
if "the_game" not in st.session_state:
    st.session_state['the_game'] = 0
# Initialization of the event variable
if "competition" not in st.session_state:
    st.session_state['competition'] = "0"
# Initialization of the event variable
if "the_vid" not in st.session_state:
    st.session_state['the_vid'] = "0"


######################################## THE LAYOUT OF THE PAGE ###########################################
#config of the page
st.set_page_config(page_title="SPORTS HIGHLIGHT GENERATOR🏀🏆", page_icon="🏀", layout="wide",
                   initial_sidebar_state="expanded", menu_items=None)

# insert empty spaces
def empty_line(lines_number):
    for num in range(lines_number):
        st.write("\n")

# set background wallpaper and subtitle title & sidebar name
def add_bg_from_url():
    st.markdown(
        f"""
       <style>
       .stApp {{
       background-image: url("https://blenderartists.org/uploads/default/original/4X/3/e/d/3ed4afe4caf93681bffb73bf21382c2fe271d141.jpg");
       background-attachment: fixed;
       background-size: cover;
       background-repeat: no-repeat;
       }}
       </style>
       """,
        unsafe_allow_html=True
    )
    add_logo("https://i0.wp.com/www.esleschool.com/wp-content/uploads/2021/03/sports-1.png?resize=120%2C120&ssl=1")
    st.sidebar.markdown("# SPORTS HIGHLIGHT GENERATOR🏀🏆")
    # set the homepage style
    empty_line(5)
    st.markdown(title, unsafe_allow_html=True)
    empty_line(4)

add_bg_from_url()

# set the homepage style
st.markdown(homepage, unsafe_allow_html=True)
empty_line(5)

################################################# CODE STUFF ######################################

# create 3 columns for each competition
st.markdown(comp, unsafe_allow_html=True)
empty_line(3)
col1, col2, col3 = st.columns(3, gap="large")


with col1:
    eurbut = st.button("Euroleague")
    # st.image("https://images.eurohoops.net/2019/05/ba5ac474-euroleague_logo-625x375.jpg")
    st.image("https://dd20lazkioz9n.cloudfront.net/wp-content/uploads/2021/06/Euroleague_Logo_Stacked.png")
    if eurbut:
        st.session_state.competition = "Euroleague"
        st.markdown("# Loading... Please wait🙂")
        switch_page("game highlights")

with col2:
   nbabut = st.button("NBA")
   # st.image("https://andscape.com/wp-content/uploads/2017/06/nbalogo.jpg?w=700")
   st.image("https://1000logos.net/wp-content/uploads/2017/04/Logo-NBA.png")

   if nbabut:
       st.session_state.competition = "Nba"
       st.sidebar.success("Not yet implemented")

with col3:
   grbut = st.button("Greek Basket League")
   #st.image("https://athlitikoskosmos.gr/wp-content/uploads/2022/10/inbound8215984157073710095.jpg")
   st.image("https://assets.b365api.com/images/wp/o/eff877d8fa1926f2f8423fa038e38f1a.png")

   if grbut:
       st.session_state.competition = "Basket League"
       st.sidebar.success("Not yet implemented")




####### ON THE FLY #########
# # game options to watch from
# my_options = ("Choose from the available Games", "CSKA Moscow Vs Barcelona", "Olympiakos Vs Panathinaikos", "CSKA Moscow Vs Bayern Munich")
# st.write("\n")
# # make a menu with selectbox
# game_vid = st.selectbox("What Game you want to watch Highlights from?", my_options, index=0, key=None,
#                         help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")
# #save game_vid value
# st.session_state.the_game = game_vid

# # if statement for the games
# if game_vid == "CSKA Moscow Vs Barcelona":
#
#         myfps = match_scl(filepaths.trim_vid_eu, filepaths.cska_barc_vid, filepaths.ocr_eur, filepaths.tmp_eu, 33.5, 34.5)        # NA DINW THN TEMP IMAGE EDW
#         st.session_state.fps = myfps
#
#         # ocr the frames matching temp with easyOcr
#         ttags, succ_r = easyOcr_dir(filepaths.ocr_eur)  # na ta kanw save kapou ta ttags         # TA TTAGS GIA KATHE MATCH ALLO FAKELO
#         st.session_state.timetags = ttags
#
# elif game_vid == "Olympiakos Vs Panathinaikos":
#
#         myfps = match_scl(filepaths.trim_vid_eu, filepaths.cska_barc_vid, filepaths.ocr_eur, filepaths.tmp_eu, 33.5, 34.5)        # NA DINW THN TEMP IMAGE EDW
#         st.session_state.fps = myfps
#
#         # ocr the frames matching temp with easyOcr
#         ttags, succ_r = easyOcr_dir(filepaths.ocr_eur)  # na ta kanw save kapou ta ttags         # TA TTAGS GIA KATHE MATCH ALLO FAKELO
#         st.session_state.timetags = ttags
#
# elif game_vid == "CSKA Moscow Vs Bayern Munich":
#
#         myfps = match_scl(filepaths.trim_vid_eu, filepaths.cska_barc_vid, filepaths.ocr_eur, filepaths.tmp_eu, 33.5, 34.5)        # NA DINW THN TEMP IMAGE EDW
#         st.session_state.fps = myfps
#
#         # ocr the frames matching temp with easyOcr
#         ttags, succ_r = easyOcr_dir(filepaths.ocr_eur)  # na ta kanw save kapou ta ttags         # TA TTAGS GIA KATHE MATCH ALLO FAKELO
#         st.session_state.timetags = ttags