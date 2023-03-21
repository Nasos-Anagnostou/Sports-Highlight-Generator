# author: Nasos Anagnostou 
# Diploma Thesis "Semantic event analysis in sports Video using webcast Text"
# Create file paths - initialise each file path easy 

# ocr needed time pattern and under minute format
time_pat = '((1[012]|0[0-9]|[0-9]):([0-9][0-9]))|(([1-5][0-9]|[0-9])(\.)[0-9])'
time_pat2 = '((1[012]|0[0-9]|[0-9])(\.|\,|\;|\:)([0-9][0-9]))|(([1-5][0-9]|[0-9])(\.|\,)[0-9])'
under_minute_format = '(([1-5][0-9]|[0-9])(\.|\,)[0-9])'

# working directory root_path
root = r"E:\Career files\Degree Thesis"  # move to aligner
root_path = r"E:\Career files\Degree Thesis\2. Dataset\competition_paths"

# roi path
roi_path = root + r"/2. Dataset/Object_Det_files/"  # move to aligner

# timetags directory
timetags = root_path + r"/timetags"

# 1. select the path extracted frames will be saved
ocr_eu1 = root_path + r"/ocr_paths/ocr_eu1/"  # dimiourgia neou xorou kathe fora? - move to aligner
ocr_eu2 = root_path + r"/ocr_paths/ocr_eu2/"
ocr_eu3 = root_path + r"/ocr_paths/ocr_eu3/"
ocr_eu4 = root_path + r"/ocr_paths/ocr_eu4/"
ocr_temp = root_path + r"/ocr_paths/ocr_temp/"
ocr_nba1 = root_path + r"/ocr_paths/ocr_nba1/"

# 2. csv files for each competition and each game
#euroleague
cska_barc_csv = root_path + r"/csv_paths/csv_eur/cska_barc.csv"
oly_pao_csv = root_path + r"/csv_paths/csv_eur/oly_pao.csv"
cska_bayern_csv = root_path + r"/csv_paths/csv_eur/cska_bayern.csv"
fener_zalgiris = root_path + r"/csv_paths/csv_eur/fener_zalgiris.csv"

# 3. input template and full game video file
cska_barc_vid = root_path + r"/game_vid/CSKA_BARCA.mp4"
oly_pao_vid = root_path + r"/game_vid/Olympiakos vs Panathinaikos_Euroleague 2021 J24.mp4"
cska_bayern_vid = root_path + r"/game_vid/Cska Moscow vs Bayern Munich Full Game Euroleague Round 23.mp4"
fener_zalgiris_vid = root_path + r"/game_vid/Fenerbahce Istanbul vs Zalgiris Kaunas EUROLEAGUE FULL GAME.mp4"
bucks_lakers_vid = root + r"/2. Dataset/game videos/Milwaukee Bucks vs Los Angeles Clippers 06.02.2022.mp4"


# 4. image template for each competition
template_root = r"E:\Career files\Degree Thesis\2. Dataset\competition_paths\tmp_path/"
tmp_eu  = root_path + r"/tmp_path/timebox_eu.jpg"
tmp_eu2  = root_path + r"/tmp_path/template_eu.jpg"
tmp_nba = root_path + r"/tmp_path/timebox_nba.jpg"
tmp_gr  = root_path + r"/tmp_path/timebox_gr.jpg"

# 5. trim X minutes from full match video and save it in the chosen dir
trim_vid_eu  = root_path + r"/trimmed_videos/"
trim_vid_eu1  = root_path + r"/trimmed_videos/" + cska_barc_vid[68: -4] +'_trimmed.mp4'
trim_vid_eu2  = root_path + r"/trimmed_videos/" + oly_pao_vid[68: -4] +'_trimmed.mp4'
trim_vid_eu3  = root_path + r"/trimmed_videos/" + cska_bayern_vid[68: -4] +'_trimmed.mp4'

trim_vid_nba = root_path + r"/trimmed_videos/trim_nba.mp4"
trim_vid_gr  = root_path + r"/trimmed_videos/trim_gr.mp4"

# chosen event video clip
clip_1 = root_path + r"/trimmed_videos/clip_1.mp4"  # move to clip_creator

