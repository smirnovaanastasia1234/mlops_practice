import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import time
from PIL import  Image

# Title of the application 
st.title(" Добро пожаловать! 👋")

st.header('  Анализ тональности сообщений Телеграм\n ')
st.info("Группа 32: Смирнова А., Кожедуб Н., Багаудинов Э., Петраков В.")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)


#загружаю модель
@st.cache

def read_data(uploaded_file):
    return pd.read_csv(uploaded_file)

datafile = st.sidebar.file_uploader("Загрузите файл csv", ["csv"])
if datafile is None:
    st.info("""Загрузите набор данных (.csv) на боковой панели, чтобы приступить к работе.""")
    st.stop()

data = read_data(datafile).copy()
dat = data.dropna(axis='index', how='any', subset=['text'])

#обучение модели
@st.cache(allow_output_mutation=True)
def load_model():

    model=pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")
    return model
                      
result = st.sidebar.button('🤗Распознать')
#Индикатор выполнения
progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")

st.balloons()

model = load_model()
df_model = dat.copy()

if result:
    lst = []
    for i in df_model["text"]:
        lst.append(model(str(i))[0]["label"])
        df_model["Sentinent"]=pd.DataFrame(lst)
        

#Это вкладки
    tab1, tab2, tab3 = st.tabs(["Загруженные данные", "Данные после обучения", "Анализ настроений"])

    tab1.subheader("Загруженные данные")
    tab1.write(data)

    tab2.subheader("Данные после обучения")

    with tab2:
        with st.expander("Все сообщения"):
            st.write(df_model)
        with st.expander("🙁 Негативные сообщения"):
            st.write(df_model[df_model["Sentinent"]=="NEGATIVE"])
        with st.expander("🙃 Позитивные  сообщения"):
            st.write(df_model[df_model["Sentinent"]=="POSITIVE"])
        with st.expander("😐 Нейтральные сообщения"):
            st.write(df_model[df_model["Sentinent"]=="NEUTRAL"])

    tab3.subheader("Анализ настроений")
    with tab3:
        st.subheader("Распределение тональности сообщений")
        col1, col2 = st.columns(2)
        with col1:
        #Первый график
            fig = plt.figure()
            palette = sns.color_palette('PiYG_r', 15)
            plt.title('Распределение настроений')
            sns.countplot(df_model['Sentinent'], palette=palette)
            st.pyplot(fig)

        with col2:
        #Второй график
            comment_words = ''

            for val in df_model.text:
                val = str(val)
                tokens = val.split()
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
            
                comment_words += " ".join(tokens)+" "

            wordcloud = WordCloud(background_color = "white",width=800, height=640).generate(comment_words)

            fig = plt.figure()
            plt.title('Облако слов')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)

        st.markdown("------------------------------------------------------------------------------------")
   
        st.subheader("Негативные сообщения")
        col3, col4 = st.columns(2)
        with col3:
            fig = plt.figure()
            df_P= df_model[df_model["Sentinent"]=="NEGATIVE"]
            df_p =df_P.user_id.value_counts().sort_index()
            df_p.plot.bar(edgecolor='k', alpha=0.9, stacked = True, cmap="hot")
            plt.title("Самые негативные user_id	")
            st.pyplot(fig)

        with col4:
            comment_words = ''
            df_P = df_model[df_model["Sentinent"]=="NEGATIVE"]
            for val in df_P.text:
                val = str(val)
                tokens = val.split()
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                
                comment_words += " ".join(tokens)+" "

            wordcloud = WordCloud(background_color = "white",width=800, height=640, colormap="YlOrRd" ).generate(comment_words)

            fig = plt.figure()
            plt.title('Облако слов негативных сообщений')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)   

        st.markdown("------------------------------------------------------------------------------------")

        st.subheader("Позитивные сообщения")
        col5, col6 = st.columns(2)
        with col5:
            fig = plt.figure()
            df_N = df_model[df_model["Sentinent"]=="POSITIVE"]
            df_n =df_N.user_id.value_counts().sort_index()
            df_n.plot.bar(edgecolor='k', alpha=0.9, stacked = True, cmap="cividis")
            plt.title("Самые позитивные user_id	")
            st.pyplot(fig)

        with col6:
            comment_words = ''
            df_N = df_model[df_model["Sentinent"]=="POSITIVE"]
            for val in df_N.text:
                val = str(val)
                tokens = val.split()
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()
                
                comment_words += " ".join(tokens)+" "

            wordcloud = WordCloud(background_color = "white",width=800, height=640, colormap="PuBu_r").generate(comment_words)

            fig = plt.figure()
            plt.title('Облако слов позитивных сообщений')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig) 
