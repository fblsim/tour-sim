from requests_html import HTMLSession
import pandas as pd

def get_ratings():
    session = HTMLSession()
    
    url = "https://www.eloratings.net/World.tsv?_=1591707820981"
     
    r = session.get(url)
    tmp = r.content
    tmp2 = tmp.decode('utf-8')
       
    res = pd.DataFrame([r.split('\t') for r in tmp2.split('\n')])

    res = res.reindex(columns = [2,3])
    
    res.columns = ['Country', 'ELO']      

    cdict = get_translation()
    
    res['Country'] = res['Country'].map(cdict)
    res['ELO'] = res['ELO'].astype('float')
    
    res = rename_countries(res)
    
    res = res.set_index('Country')
    return res

def get_translation():     
    session = HTMLSession()        
    url2 = "https://www.eloratings.net/en.teams.tsv?_=1591707820979"
    r = session.get(url2)
    tmp = r.content
    
    tmp2 = tmp.decode('utf-8')
    res = pd.DataFrame([r.split('\t') for r in tmp2.split('\n')])
    
    
    res = res.reindex(columns = [0,1]).set_index(0)[1]             
    
    return res.to_dict()



def rename_countries(df_elo):
    
    trans = {'Czechia' : 'Czech Republic'}
    df_elo['Country'] = df_elo['Country'].replace(trans)
    
    return df_elo