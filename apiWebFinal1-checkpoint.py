import streamlit as st
import pandas as pd
import numpy as np
import datetime
# from random import *
from numpy.random import randint
from PIL import Image as img

def intro():
    import streamlit as st

    st.write("# Welcome PROJET 7 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement
        
        ## Utilisation
        
        Menu sur votre gauche
        
        ### 2 types de saisies :

        - Form : Mode Curseurs : Saisie par curseurs
        - Form : Mode Sasie : Saisie Manuelle

        ### Plus d'infos :

        - Statistiques : Satistiques des données connues (Min max Q1 / Q3)
        - Exmples : 50 lihnes de donnees exemples
        - Model Affichage du modele utilisé
    """
    )

def nyx_demo(nbVal=22):
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:silver;padding:3px">
    <h2 style ="color:black;text-align:center;">Streamlit NYX : Mode saisie </h2>
    </div>    
    """
    st.markdown(html_temp, unsafe_allow_html = True)  
    # st.markdown(nbVal, unsafe_allow_html = True)
    
    from datetime import datetime
    def convert(date_time) :
        f='%Y-%m-%d'
        datetime_str = datetime.strptime(date_time,f)
        return datetime_str
    
    import matplotlib.pyplot as plt
    
    # GRAPHIQUE / VARIABLE
    def boxplotNyx2 (df,val) : 
        labels = df.columns.tolist()
        titr=labels[0][0:-2]
        q1=df.iloc[2, 1]
        q3=df.iloc[3, 1]
        if val>q1 and val<q3 :
            err = 0
            backColor='lightgreen'
            linColor='darkgreen'
        else : 
            backColor='pink'
            linColor='red'      
        centr = ((q3-q1)/2)
        demiDelta = (q3-centr)
        err = (((val-centr)/demiDelta)-1)*100
        st.markdown(titr)
        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})

        fig, ax = plt.subplots(1,1,figsize=(13,5),facecolor='silver')
        ax.set_title(titr+' > '+str(val))
        ax.axvline(x=val,color=linColor)
        #    ax.layout=None
        #    ax.c='white'
        ax.set_facecolor(backColor)
        ax.bxp(bxp_stats, showfliers=False, vert=False,)

        barplot_chart = st.write(fig)
  
        return err
    
    # IMAGE
    from PIL import Image

         
    # IMPORT MODEL
    # import lightgbm as lgb
    import joblib
    with open('P7/modem_score_lg.pkl', 'rb') as f:
        model=joblib.load(f)
 
     
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    
    # IMPORT DES STATS
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    
    # LIGNE AU HASARD 
    lign=randint(1, dfExempleSize)-1  
    # lign=1
    SK_ID_CURR=dfExemple.loc[lign,'SK_ID_CURR']
    dflign=dfExemple[dfExemple['SK_ID_CURR']==SK_ID_CURR]
    

    # st.markdown(str(model)+' >> '+str(dfExempleSize)+' ['+str(lign)+']', unsafe_allow_html = True)  
    
    # --------------------------------------------- FORMULAIRE
    # 'SK_ID_CURR', 
    # 'PAYMENT_RATE', 
    # 'EXT_SOURCE_1', 
    # 'EXT_SOURCE_3', 
    # 'EXT_SOURCE_2', 
    # 'DAYS_BIRTH', 
    # 'AMT_ANNUITY', 
    # 'DAYS_EMPLOYED', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 
    # 'ACTIVE_DAYS_CREDIT_MAX', 
    # 'APPROVED_CNT_PAYMENT_MEAN', 
    # 'PREV_CNT_PAYMENT_MEAN', 
    # 'INSTAL_DPD_MEAN', 
    # 'DAYS_ID_PUBLISH', 
    # 'INSTAL_AMT_PAYMENT_MIN', 
    # 'ANNUITY_INCOME_PERC', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 
    # 'DAYS_EMPLOYED_PERC', 
    # 'AMT_CREDIT', 
    # 'POS_MONTHS_BALANCE_SIZE', 
    # 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 
    # 'AMT_GOODS_PRICE', 
    # 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 
    # 'INSTAL_AMT_PAYMENT_SUM', 
    # 'OWN_CAR_AGE', 
    # 'DAYS_REGISTRATION'

    listvardays=('DAYS_BIRTH', 'DAYS_EMPLOYED', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'DAYS_ID_PUBLISH', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'DAYS_EMPLOYED_PERC' ,'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'DAYS_REGISTRATION')
    
    i=0
    j=0
    listVar=[]
    initial_date = "12/1/2022"
    initial_date2 = convert("2022-12-01")
    f='%Y-%m-%d'   
    dfConvert = pd.DataFrame(columns=['coef','decal'])  
    echel=1000
    with st.form("Predict_form"):
        for c in dflign :
            var=c
            listVar.append(c)
            val = dflign.iloc[0,i]
            if c != 'SK_ID_CURR' and i < nbVal :
                if c in listvardays :
                    req_date = pd.to_datetime(initial_date) + pd.DateOffset(days=round(val))
                    globals()[var] = st.date_input(c, req_date)
                else :
                    tmpminmax = dfStat[[var+'_0',var+'_1']].filter(items = ['min','max'], axis=0).stack()
                    mini=tmpminmax.min()
                    maxi=tmpminmax.max()
                    txtTitr=c+' ['+str(mini)+' ; '+str(maxi)+'] '+str(val)
                    # st.markdown(txtTitr, unsafe_allow_html = True)
                    globals()[var] = st.text_input(txtTitr,  value=val) #step=1, min_value=mini, max_value=maxi,
                    # st.markdown(coef, unsafe_allow_html = True)
                    j+=1
            else : 
                globals()[var] = val                
            i+=1
        submitted = st.form_submit_button("Submit")       
        # st.markdown('         [l: '+str(lign)+' / '+str(dfExempleSize)+']', unsafe_allow_html = True)  
        i=0
        j=0
        if submitted: 
            # st.markdown(listVar , unsafe_allow_html = True)
            tabinfo=[]
            for c in listVar :
                if c != 'SK_ID_CURR' and i < nbVal :
                    if c in listvardays :
                        # st.markdown('DATE', unsafe_allow_html = True)
                        lifetime= convert(str(globals()[c])) - initial_date2
                        # st.markdown(lifetime.days , unsafe_allow_html = True)
                        tabinfo.append(lifetime.days)
                    else :
                        tabinfo.append(float(globals()[c]))
                        j+=1
                else : 
                    tabinfo.append(globals()[c])
                i+=1
            st.markdown(tabinfo, unsafe_allow_html = True) 
            info = pd.DataFrame([tabinfo], columns = listVar)
            result = model.predict(info)
            st.sidebar.markdown('''
                # 
                - [Verdict](#section-1)
                - [Détail / varialble](#section-2)
                - [SYNTHESE](#section-3)
                ''', unsafe_allow_html=True)

            st.header('Section 1')
            st.title("VERDICT")
            st.markdown('The output is {}'.format(result), unsafe_allow_html = True) 
            if result > 0.5 :
                image = Image.open('P7/Img/OK.png')
                synthColor='lightgreen'
            else : 
                image = Image.open('P7/Img/KO.png') 
                synthColor='pink'
            st.image(image, caption='Sunrise by the mountains')
            # st.markdown(dflign.shape, unsafe_allow_html = True) 
        
            st.header('Section 2')
            st.title("Détail / variable")
            # GRAPHIQUES PAR VARAIBLES
            i=1
            err=0
            dfSynthes = pd.DataFrame(columns=['var','coef']) 
            list = dfStat.columns.tolist()[::2]
            for v in list :
                if v != 'OWN_CAR_AGE' and i < (nbVal-1) :
                    v2=v[0:-1]+'1'
                    varval=v[0:-2]
                    print (v,' & ',v2)    
                    err = boxplotNyx2 (dfStat[[v,v2]],tabinfo[i])
                    dfSynthes.loc[i] = [varval,err]
                i+=1
             
            st.header('Section 3')
            st.title("Synthèse")
            # SYNTHESE 
            fig = plt.figure()
            plt.style.use('fivethirtyeight')
            group_data = dfSynthes['coef'].tolist()
            group_names = dfSynthes['var'].tolist()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.barh(group_names, group_data)
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=45, horizontalalignment='right')
            ax.axvline(0, ls='-', color='r')
            ax.set(xlabel='Ratio Q3/Q1', ylabel='Paramètre',title='Synthese')
            ax.set_facecolor(synthColor)
            st.write(fig)
    
def nyx_demo2(nbVal=22):
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:silver;padding:3px">
    <h2 style ="color:black;text-align:center;">Streamlit NYX : Mode Curseurs </h2>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)  
    # st.markdown(nbVal, unsafe_allow_html = True.
    
    from datetime import datetime
    def convert(date_time) :
        f='%Y-%m-%d'
        datetime_str = datetime.strptime(date_time,f)
        return datetime_str
    
    import matplotlib.pyplot as plt
    
    def boxplotNyx2 (df,val) : 
        labels = df.columns.tolist()
        titr=labels[0][0:-2]
        q1=df.iloc[2, 1]
        q3=df.iloc[3, 1]
        if val>q1 and val<q3 :
            err = 0
            backColor='lightgreen'
            linColor='darkgreen'
        else : 
            backColor='pink'
            linColor='red'      
        centr = ((q3-q1)/2)
        demiDelta = (q3-centr)
        err = (((val-centr)/demiDelta)-1)*100
        st.markdown(titr)
        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})

        fig, ax = plt.subplots(1,1,figsize=(13,5),facecolor='silver')
        ax.set_title(titr+' > '+str(val))
        ax.axvline(x=val,color=linColor)
        #    ax.layout=None
        #    ax.c='white'
        ax.set_facecolor(backColor)
        ax.bxp(bxp_stats, showfliers=False, vert=False,)

        barplot_chart = st.write(fig)
  
        return err
    
    # IMAGE
    from PIL import Image

         
    # IMPORT MODEL
    # import lightgbm as lgb
    import joblib
    with open('P7/modem_score_lg.pkl', 'rb') as f:
        model=joblib.load(f)
 
     
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    
    # IMPORT DES STATS
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    
    # LIGNE AU HASARD 
    lign=randint(1, dfExempleSize)-1  
    # lign=1
    SK_ID_CURR=dfExemple.loc[lign,'SK_ID_CURR']
    dflign=dfExemple[dfExemple['SK_ID_CURR']==SK_ID_CURR]
    
    # st.markdown(str(dfExempleSize)+' ['+str(lign)+']', unsafe_allow_html = True)  
    # st.markdown(str(model)+' >> '+str(dfExempleSize)+' ['+str(lign)+']', unsafe_allow_html = True)  
    
    # --------------------------------------------- FORMULAIRE
    # 'SK_ID_CURR', 
    # 'PAYMENT_RATE', 
    # 'EXT_SOURCE_1', 
    # 'EXT_SOURCE_3', 
    # 'EXT_SOURCE_2', 
    # 'DAYS_BIRTH', 
    # 'AMT_ANNUITY', 
    # 'DAYS_EMPLOYED', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 
    # 'ACTIVE_DAYS_CREDIT_MAX', 
    # 'APPROVED_CNT_PAYMENT_MEAN', 
    # 'PREV_CNT_PAYMENT_MEAN', 
    # 'INSTAL_DPD_MEAN', 
    # 'DAYS_ID_PUBLISH', 
    # 'INSTAL_AMT_PAYMENT_MIN', 
    # 'ANNUITY_INCOME_PERC', 
    # 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 
    # 'DAYS_EMPLOYED_PERC', 
    # 'AMT_CREDIT', 
    # 'POS_MONTHS_BALANCE_SIZE', 
    # 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 
    # 'AMT_GOODS_PRICE', 
    # 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 
    # 'INSTAL_AMT_PAYMENT_SUM', 
    # 'OWN_CAR_AGE', 
    # 'DAYS_REGISTRATION'

    listvardays=('DAYS_BIRTH', 'DAYS_EMPLOYED', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'DAYS_ID_PUBLISH', 'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'DAYS_EMPLOYED_PERC' ,'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'DAYS_REGISTRATION')
    
    i=0
    j=0
    listVar=[]
    initial_date = "12/1/2022"
    initial_date2 = convert("2022-12-01")
    f='%Y-%m-%d'   
    dfConvert = pd.DataFrame(columns=['coef','decal'])  
    echel=1000
    with st.form("Predict_form"):
        for c in dflign :
            var=c
            listVar.append(c)
            val = dflign.iloc[0,i]
            if c != 'SK_ID_CURR' and i < nbVal :
                if c in listvardays :
                    req_date = pd.to_datetime(initial_date) + pd.DateOffset(days=round(val))
                    globals()[var] = st.date_input(c, req_date)
                else :
                    tmpminmax = dfStat[[var+'_0',var+'_1']].filter(items = ['min','max'], axis=0).stack()
                    mini=tmpminmax.min()
                    maxi=tmpminmax.max()
                    coef=(maxi-mini)/echel
                    dfConvert.loc[j] = [coef,mini]
                    valConvert=round((val-mini)/coef)
                    txtTitr=c+' ['+str(mini)+' ; '+str(maxi)+'] '+str(val)+' >> '+str(valConvert)
                    # st.markdown(txtTitr, unsafe_allow_html = True)
                    globals()[var] = st.slider(txtTitr, min_value=0, max_value=echel, step=1, value=valConvert)
                    # st.markdown(coef, unsafe_allow_html = True)
                    j+=1
            else : 
                globals()[var] = val                
            i+=1
        # st.markdown(dfConvert, unsafe_allow_html = True)     
        submitted = st.form_submit_button("Submit")    
        i=0
        j=0
        if submitted: 
            # st.markdown(listVar , unsafe_allow_html = True)
            tabinfo=[]
            for c in listVar :
                if c != 'SK_ID_CURR' and i < nbVal :
                    # st.markdown(c, unsafe_allow_html = True)
                    # st.markdown(globals()[c], unsafe_allow_html = True) 
                    if c in listvardays :
                        # st.markdown('DATE', unsafe_allow_html = True)
                        lifetime= convert(str(globals()[c])) - initial_date2
                        # st.markdown(lifetime.days , unsafe_allow_html = True)
                        tabinfo.append(lifetime.days)
                    else :
                        valConvert=globals()[c]
                        valForm=dfConvert.iloc[j, 1]+(valConvert*dfConvert.iloc[j, 0])
                        tabinfo.append(valForm)
                        # st.markdown(str(c)+' : '+str(valForm)+' = '+str(dfConvert.iloc[j, 1])+' + ( '+str(valConvert)+' * '+str(dfConvert.iloc[j, 0])+')', unsafe_allow_html = True)
                        j+=1
                else : 
                    tabinfo.append(globals()[c])
                i+=1
            # st.markdown(tabinfo, unsafe_allow_html = True) 
            info = pd.DataFrame([tabinfo], columns = listVar)
            result = model.predict(info)
            # st.success('The output is {}'.format(result))
            st.sidebar.markdown('''
                # 
                - [Verdict](#section-1)
                - [Détail / varialble](#section-2)
                - [SYNTHESE](#section-3)
                ''', unsafe_allow_html=True)

            st.header('Section 1')
            st.title("VERDICT")
            st.markdown('The output is {}'.format(result), unsafe_allow_html = True) 
            if result > 0.5 :
                image = Image.open('P7/Img/OK.png')
                synthColor='lightgreen'
            else : 
                image = Image.open('P7/Img/KO.png') 
                synthColor='pink'
            st.image(image, caption='Sunrise by the mountains')
            # st.markdown(dflign.shape, unsafe_allow_html = True) 
        
            st.header('Section 2')
            st.title("Détail / variable")
            # GRAPHIQUES PAR VARAIBLES
            i=1
            err=0
            dfSynthes = pd.DataFrame(columns=['var','coef']) 
            list = dfStat.columns.tolist()[::2]
            for v in list :
                if v != 'OWN_CAR_AGE' and i < (nbVal-1) :
                    v2=v[0:-1]+'1'
                    varval=v[0:-2]
                    print (v,' & ',v2)    
                    err = boxplotNyx2 (dfStat[[v,v2]],tabinfo[i])
                    dfSynthes.loc[i] = [varval,err]
                i+=1
             
            st.header('Section 3')
            st.title("Synthèse")
            # SYNTHESE 
            fig = plt.figure()
            plt.style.use('fivethirtyeight')
            group_data = dfSynthes['coef'].tolist()
            group_names = dfSynthes['var'].tolist()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.barh(group_names, group_data)
            labels = ax.get_xticklabels()
            plt.setp(labels, rotation=45, horizontalalignment='right')
            ax.axvline(0, ls='-', color='r')
            ax.set(xlabel='Ratio Q3/Q1', ylabel='Paramètre',title='Synthese')
            ax.set_facecolor(synthColor)
            st.write(fig)

def nyx_df():
    
     ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:silver;padding:3px">
    <h2 style ="color:black;text-align:center;">Streamlit NYX : Datas exemples </h2>
    </div>    
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)   

    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    
    # IMPORT DATA EXEMPLES
    dfExemple=pd.read_csv('P7/X4Sample.csv', sep=';')
    dfExempleSize=dfExemple.shape[0]
    st.markdown(dfExempleSize, unsafe_allow_html = True)   
    
    st.dataframe(dfExemple, use_container_width=st.session_state.use_container_width)
    


def nyx_stat():
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:silver;padding:3px">
    <h2 style ="color:black;text-align:center;">Streamlit NYX : Statistiques </h2>
    </div>    
    """
 
    
    import matplotlib.pyplot as plt
    def boxplotNyx (df) : 
        labels = df.columns.tolist()

        # create dictionaries for each column as items of a list
        bxp_stats = df.apply(lambda x: {'med':x.med, 'q1':x.q1, 'q3':x.q3, 'whislo':x['min'], 'whishi':x['max']}, axis=0).tolist()

        # add the column names as labels to each dictionary entry
        for index, item in enumerate(bxp_stats):
            item.update({'label':labels[index]})

        # plt.subplots(1,1,figsize=(13,5),facecolor='silver')
        fig, ax = plt.subplots(1,1,figsize=(13,5),facecolor='silver')
        # plt.figure(figsize=(15,3))
        ax.set_title(labels[0][0:-2])
        ax.bxp(bxp_stats, showfliers=False, vert=False,);
        # plt.show()
        barplot_chart = st.write(fig)
    
    st.markdown(html_temp, unsafe_allow_html = True)  
    
    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")
    
    dfStat=pd.read_csv('P7/df_BarplotInfo.csv', sep=';')
    dfStat.index = ['max','min','q1','q3','med']
    st.dataframe(dfStat, use_container_width=st.session_state.use_container_width)
    
    list = dfStat.columns.tolist()[::2]
    for v in list :
        if v != 'OWN_CAR_AGE_0'  :
            v2=v[0:-1]+'1'
            print (v,' & ',v2)    
            boxplotNyx (dfStat[[v,v2]])

            
def nyx_model():
    
    ## giving the webpage a title
    st.title("Prêt à dépenser")
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:silver;padding:3px">
    <h2 style ="color:black;text-align:center;">Streamlit NYX : MOPDELE </h2>
    </div>    
    """
    # IMPORT MODEL modem_score_lg
    import joblib
    # import lightgbm as lgb   
    with open('P7/modem_score_lg.pkl', 'rb') as f:
        model=joblib.load(f)
    
    st.markdown(str(model), unsafe_allow_html = True)  

    
page_names_to_funcs = {
    "—": intro,
    "Form : Mode Curseurs": nyx_demo2,
    "Form : Mode Sasie": nyx_demo,
    "Statistiques": nyx_stat,
    "Exmples": nyx_df,
    "Model": nyx_model
}

demo_name = st.sidebar.selectbox("Menu (v6)", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()