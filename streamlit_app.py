import streamlit as st
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

import sqlite3
import datetime

zaman=str(datetime.datetime.now())
conn=sqlite3.connect("trendyorum.sqlite3")
c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS testler(yorum TEXT, sonuc TEXT, zaman TEXT)")
conn.commit()

cv=CountVectorizer(max_features=300)
rf=RandomForestClassifier()

df=pd.read_csv("yorum.csv.zip",on_bad_lines="skip",delimiter=";")

def temizle(sutun):
    stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
    semboller=string.punctuation
    sutun=sutun.lower()
    for se in semboller:
        sutun=sutun.replace(se," ")
    for st in stopwords:
        ss=" "+st+" "
        sutun=sutun.replace(ss," ")
    sutun=sutun.replace("  "," ")
    return sutun


df["Metin"]=df["Metin"].apply(temizle)

X=cv.fit_transform(df["Metin"]).toarray();
y=df["Durum"]

x_train,x_test,y_train,y_test=tts(X,y,random_state=42,train_size=0.75)

st.header("E-ticaret Yorum Değerlendirme Uygulaması")
yorum=st.text_area("Yorum metnini giriniz")
btn=st.button("Yorumu kategorilendir")

if btn:
    model=rf.fit(x_train,y_train)
    skor=model.score(x_test,y_test)
    tahmincumle=temizle(yorum)
    tahmin=cv.transform(np.array([tahmincumle])).toarray()
    kat={
        1:"Olumlu",
        0:"Olumsuz",
        2:"Nötr"
    }
    sonuc=model.predict(tahmin)
    s=kat.get(sonuc[0])
    st.subheader(s)
    st.write("Model skoru :",skor)
    c.execute("INSERT INTO testler VALUES(?,?,?)",(yorum,s,zaman))
    conn.commit()

c.execute("SELECT * FROM testler")
testler=c.fetchall()
st.table(testler)


kod="""
import streamlit as st
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier

import sqlite3
import datetime

zaman=str(datetime.datetime.now())
conn=sqlite3.connect("trendyorum.sqlite3")
c=conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS testler(yorum TEXT, sonuc TEXT, zaman TEXT)")
conn.commit()

cv=CountVectorizer(max_features=300)
rf=RandomForestClassifier()

df=pd.read_csv("yorum.csv.zip",on_bad_lines="skip",delimiter=";")

def temizle(sutun):
    stopwords=['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
    semboller=string.punctuation
    sutun=sutun.lower()
    for se in semboller:
        sutun=sutun.replace(se," ")
    for st in stopwords:
        ss=" "+st+" "
        sutun=sutun.replace(ss," ")
    sutun=sutun.replace("  "," ")
    return sutun


df["Metin"]=df["Metin"].apply(temizle)

X=cv.fit_transform(df["Metin"]).toarray();
y=df["Durum"]

x_train,x_test,y_train,y_test=tts(X,y,random_state=42,train_size=0.75)

st.header("E-ticaret Yorum Değerlendirme Uygulaması")
yorum=st.text_area("Yorum metnini giriniz")
btn=st.button("Yorumu kategorilendir")

if btn:
    model=rf.fit(x_train,y_train)
    skor=model.score(x_test,y_test)
    tahmincumle=temizle(yorum)
    tahmin=cv.transform(np.array([tahmincumle])).toarray()
    kat={
        1:"Olumlu",
        0:"Olumsuz",
        2:"Nötr"
    }
    sonuc=model.predict(tahmin)
    s=kat.get(sonuc[0])
    st.subheader(s)
    st.write("Model skoru :",skor)
    c.execute("INSERT INTO testler VALUES(?,?,?)",(yorum,s,zaman))
    conn.commit()
    
c.execute("SELECT * FROM testler")
testler=c.fetchall()
st.table(testler)
"""
with st.expander("Kaynak Kodları"):
    st.header("kaynak kodları")
    st.code(kod,language="python")



