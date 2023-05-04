import math
import random
from collections import Counter
import pandas as pd
import numpy as np
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


@st.cache_resource
def load_model():
    embedding = Wine_Embedding(embed_size=32)
    encoder = Att_Encoder(embedding=embedding, x_dim=32, y_dim=256, dropout=0.15)
    model = Trainer(encoder, embed_size=32)
    model.load_state_dict(torch.load('wine_model.pt'))
    return model


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Wine_Embedding(nn.Module):
    def __init__(self, embed_size=32):
        super(Wine_Embedding, self).__init__()
        self.tastes_list = np.array(['Bold', 'Tannic', 'Sweet', 'Acidic'])
        self.aroma_list = np.array(['Black Fruit', 'Oaky', 'Earthy', 'Spices', 'Red Fruit', 'Tree Fruit',
                                    'Dried Fruit', 'Microbio', 'Citrus', 'Ageing', 'Tropical', 'Vegetal', 'Floral'])
        self.types_list = ['Red', 'White'] 
        self.grapes_list = np.array(['Aglianico', 'Alicante Bouschet', 'Arneis', 'Baco Noir',
                             'Grenache', 'Grenache Blanc', 'Gros Manseng', 'Grüner Veltliner',
                             'Barbera', 'Blauburger', 'Blaufränkisch', 'Bobal', 'Bonarda', 
                             'Cabernet Franc', 'Cabernet Jura', 'Cabernet Sauvignon',
                             'Carignan', 'Cariñena', 'Carménère', 'Carricante', 'Castelao', 
                             'Chambourcin', 'Chardonnay', 'Cinsault',
                             'Corvina', 'Côt', 'Dakapo', 'Dolcetto', 'Frappato', 'Gaglioppo', 'Gamay',
                             'Drupeggio', 'Durif', 'Eger', 'Feteasca Neagra',  
                             'Garnacha', 'Garnacha Tinta', 'Gewürztraminer',
                             'Graciano', 'Grauvernatsch', 'Grecanico', 'Grechetto', 'Greco', 'Greco Bianco', 
                             'Inzolia', 'Kadarka', 'Kékfrankos', 'Lodi', 'Malbec', 'Malvasia',
                             'Marechal Foch', 'Marsanne', 'Marselan', 'Mataro',
                             'Mazuelo', 'Mencia', 'Merlot', 'Molinara', 'Monastrell', 'Montepulciano', 'Moristel',
                             'Moscato', 'Moscato Bianco', 'Mourvedre', 'Muscadelle', 
                             'Muscat Blanc', 'Muscat Blanc à Petits Grains',
                             'Nebbiolo', 'Nerello Mascalese', "Nero d'Avola", 'Neuburger', 
                             'Petit Courbu', 'Petit Manseng', 'Petit Verdot', 
                             'Petite Sirah', 'Piedirosso', 'Pinot Blanc', 'Pinot Grigio', 
                             'Pinot Gris', 'Pinot Noir', 'Pinotage', 'Primitivo', 'Procanico',
                             'Riesling', 'Rondinella', 'Rotgipfler', 'Roussanne', 'Sangiovese', 'Sauvignon Blanc',
                             'Sauvignon Gris', 'Shiraz/Syrah', 'Spätburgunder', 'Sylvaner', 
                             'Sémillon', 'Tannat', 'Tempranillo', 'Teroldego',
                             'Tinta Barroca', 'Tinta Roriz', 'Tinta de toro', 
                             'Tinta del Pais', 'Tinto Fino', 'Touriga Franca',
                             'Touriga Nacional', 'Trebbiano', 'Turan', 'Verdelho',
                             'Verdicchio', 'Vermentino', 'Vernatsch',
                             'Viognier', 'Viura', 'Waiheke Island', 'Weissburgunder', 
                             'Welschriesling', 'Zierfandler', 'Zinfandel', 'Zweigelt'])
        self.countries_list = np.array(['Argentina', 'Australia', 'Austria', 'Canada', 
                                        'Chile', 'France', 'Germany', 'Hungary', 
                                        'Israel', 'Italy', 'Moldova', 'New Zealand', 'Portugal', 
                                        'Romania', 'South Africa', 'Spain', 'United States'], dtype=object)
        self.grapes_list.sort()
        self.countries_list.sort()
        self.aroma_list.sort()
        
        self.grapes_list = np.concatenate([np.array(['PAD']), self.grapes_list])
        
        self.country_size = len(self.countries_list)
        self.types_size = len(self.types_list)
        self.tastes_size = len(self.tastes_list)
        self.grapes_size = len(self.grapes_list)
        self.aroma_size = len(self.aroma_list)
        
        self.country_embedding = nn.Embedding(self.country_size, embed_size)
        self.type_embedding = nn.Embedding(self.types_size, embed_size)
        self.taste_embedding = nn.Embedding(self.tastes_size, embed_size)
        self.grapes_embedding = nn.Embedding(self.grapes_size, embed_size)
        self.aroma_embedding = nn.Embedding(self.aroma_size, embed_size)
        
        self.embedding_dim = embed_size
        self.eps = 1e-12
        
    def forward(self, types=None, countries=None, grapes=None, grapes_scales=None,
                aromas=None, aromas_scales=None, tastes=None, tastes_scales=None):
        x_countries = self.country_embedding(countries)
        x_types = self.type_embedding(types)
        x_tastes = self.taste_embedding(tastes)
        x_grapes = self.grapes_embedding(grapes)
        x_aromas = self.aroma_embedding(aromas)
        
        x_grapes = self.unit(x_grapes)
        x_grapes *= grapes_scales.unsqueeze(-1)
        
        x_tastes = self.unit(x_tastes)
        x_tastes *= tastes_scales.unsqueeze(-1)
        
        x_aromas = self.unit(x_aromas)
        x_aromas *= aromas_scales.unsqueeze(-1)
        
        return x_countries, x_types, x_tastes,  x_grapes, x_aromas
        
    def unit(self,x):
        return (x + self.eps) / (torch.norm(x, dim=2).unsqueeze(-1) + self.eps)


class Att_Encoder(nn.Module):
    def __init__(self, embedding=None, x_dim=32, y_dim=256, dropout=0.15):
        super(Att_Encoder, self).__init__()
        self.embedding = embedding
        
        self.key_fc = nn.Linear(x_dim, y_dim)
        self.query_fc = nn.Linear(x_dim, y_dim)
        self.value_fc = nn.Linear(x_dim, y_dim)
        self.layer_norm = nn.LayerNorm(y_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
    
    def encoding(self, x):
        key = self.key_fc(x) # K
        query = self.query_fc(x) # Q
        value = self.value_fc(x) # V
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        att = self.softmax(scores)
        
        y = torch.matmul(att, value)
        y = self.layer_norm(y) + value
        y = self.dropout(y)
        
        return y
    
    def forward(self, types=None, countries=None, grapes=None, grapes_scales=None,
                aromas=None, aromas_scales=None, tastes=None, tastes_scales=None):
        x_countries, x_types, x_tastes, x_grapes, x_aromas = self.embedding(types=types, countries=countries, 
                                                                            grapes=grapes, grapes_scales=grapes_scales, 
                                                                            aromas=aromas,aromas_scales=aromas_scales, 
                                                                            tastes=tastes, tastes_scales=tastes_scales)

        x_countries = x_countries.unsqueeze(1)
        x_types = x_types.unsqueeze(1)
        
        x = torch.cat([x_types, x_tastes, x_grapes, x_aromas], dim=1)
        x = self.encoding(x)
        return x

    
class Trainer(nn.Module):
    def __init__(self, encoder=None, y_dim=256, dropout=0.15, embed_size=32):
        super(Trainer, self).__init__()
        self.encoder = encoder
        
        self.y_dim = y_dim
        self.dropout = dropout
        self.hidden_dim = int(self.y_dim / 2)
        
        self.common_fc = nn.Linear(self.y_dim, self.hidden_dim)
        self.activation = nn.ELU()
        self.sig = torch.sigmoid
        self.country_classifier = nn.Linear(self.hidden_dim, self.encoder.embedding.country_size)
        
        self.type_classifier = nn.Linear(self.hidden_dim, 1)
        self.taste_regressor = nn.Linear(self.hidden_dim, self.encoder.embedding.tastes_size)
        
        self.aroma_regressor = nn.Linear(self.hidden_dim, embed_size)
        self.grape_classifier = nn.Linear(self.hidden_dim, embed_size)
        
    def get_grapes(self, y_grape, grapes=None, grapes_scales=None):
        embedding_matrix = self.encoder.embedding.grapes_embedding.weight.clone()
        final_pred_y = []
        true_y = []
        
        pred_y = torch.matmul(y_grape, embedding_matrix.T)
        
        for i in range(pred_y.shape[0]):
            final_pred_y.append(pred_y[i][grapes[i][grapes[i]!=0]])
            true_y.append(grapes_scales[i][grapes_scales[i]!=0])
            
        final_pred_y = self.sig(torch.cat(final_pred_y))
        true_y = torch.cat(true_y)
        
        return final_pred_y, true_y
        
    def get_aromas(self, y_aromas, aromas=None, aromas_scales=None):
        embedding_matrix = self.encoder.embedding.aroma_embedding.weight.clone()
        
        final_pred_y = []
        true_y = []
        
        pred_y = torch.matmul(y_aromas,embedding_matrix.T)
        
        for i in range(pred_y.shape[0]):
            aromas_id = aromas[i][aromas_scales[i]!=0] 
            final_pred_y.append(pred_y[i][aromas_id])
            true_y.append(aromas_scales[i][aromas_id])
            
        final_pred_y = self.sig(torch.cat(final_pred_y))
        true_y = torch.cat(true_y)
        
        return final_pred_y, true_y
    
    def forward(self,types=None, countries=None, aromas=None,
                     aromas_scales=None, tastes=None, tastes_scales=None,
                     grapes=None,  grapes_scales=None):
        
        x = self.encoder(types=types, countries=countries, 
                        grapes=grapes, grapes_scales=grapes_scales,
                        aromas=aromas, aromas_scales=aromas_scales, 
                        tastes=tastes, tastes_scales=tastes_scales)
        
        x, _ = torch.max(x, dim= 1)
        
        y = self.activation(self.common_fc(x))
        y_country = self.country_classifier(y)
        
        y_type = self.sig(self.type_classifier(y))
        
        y_grape = self.grape_classifier(y)
        y_aromas = self.aroma_regressor(y)
        
        y_tastes = self.sig(self.taste_regressor(y))
        
        grape_pred_y, grape_true_y = self.get_grapes(y_grape, grapes,  grapes_scales)
        aroma_pred_y, aroma_true_y = self.get_aromas(y_aromas, aromas, aromas_scales)
        
        return y_country, y_type, y_tastes, grape_pred_y, grape_true_y, aroma_pred_y, aroma_true_y


def get_initial_vec(model, df_wine, wine_type, bold, tannic, sweet, acidic):
    # user wine type
    wine_type = 0 if wine_type == 'White' else 1
    
    # get user taste ['Bold', 'Tannic', 'Sweet', 'Acidic']
    my_taste = [bold, tannic, sweet, acidic]
    
    # get wine taste ['Bold', 'Tannic', 'Sweet', 'Acidic']
    df_embedding = df_wine[['embeddings']]
    all_vecs = df_wine[['bold', 'tannic', 'sweet', 'acidic']].fillna(0)
    
    # get l2 norm of wine taste with my taste
    dist = np.sum((all_vecs.values - my_taste)**2, axis=1)**(1/2)
    df = pd.DataFrame(data=dist, index=all_vecs.index, columns=['dist']).sort_values(by='dist').drop_duplicates()
    top50 = df.iloc[:50].index
    
    # get average representation of 10 random vectors. Iterate over 50 times.
    random_vecs = []
    for i in range(50):
        random_items = random.sample(top50.tolist(), 10)
        random_vecs.append(np.mean(df_embedding.iloc[random_items].values, axis=0)[0])
    
    # make it into torch.float32 from np.float32
    random_vecs = np.stack(random_vecs)
    random_vecs = torch.from_numpy(random_vecs)
    
    # predict taste
    y = model.common_fc(random_vecs.float())
    y_tastes = model.sig(model.taste_regressor(y)).cpu().detach().numpy()
    
    # return predicted representation
    min_index = np.argmin(np.sum((y_tastes - my_taste)**2,axis=1)**(1/2))
    my_vec = random_vecs[min_index].cpu().detach().numpy()
    return my_vec


@st.cache_data
def recommend_wine(df_embedding, my_vec):
    dist = np.sum((df_embedding - my_vec)**2, axis=1)**(1/2)
    dist = pd.DataFrame(dist, columns=['dist']).sort_values(by='dist')
    return dist


@st.cache_data
def best_grapes(best_df_dataset, _embedding):
    grapes1 = _embedding.grapes_list[best_df_dataset.type1]
    grapes2 = _embedding.grapes_list[best_df_dataset.type2]
    grapes3 = _embedding.grapes_list[best_df_dataset.type3]
    grapes4 = _embedding.grapes_list[best_df_dataset.type4]
    grapes5 = _embedding.grapes_list[best_df_dataset.type5]
    grapes6 = _embedding.grapes_list[best_df_dataset.type6]
    grapes7 = _embedding.grapes_list[best_df_dataset.type7]
    grapes8 = _embedding.grapes_list[best_df_dataset.type8]
    grapes = np.concatenate([grapes1,grapes2,grapes3,grapes4,grapes5,grapes6,grapes7,grapes8])
    
    top_grapes = [i[0] for i in Counter(grapes).most_common()]
    
    try:
        top_grapes.remove('PAD')
    except:
        pass
        
    return top_grapes


@st.cache_data
def best_countries(best_df_dataset, _embedding):
    countries = best_df_dataset['country']
    top_countries = [i[0] for i in Counter(countries).most_common()]
    
    return top_countries


@st.cache_data
def best_continent(best_df_dataset, _embedding):
    continents = best_df_dataset['continent']
    top_continents = [i[0] for i in Counter(continents).most_common()]
    
    return top_continents


@st.cache_data
def update_my_vec(my_vec, target_wine_vec, rate):
    my_vec = (my_vec *(2-rate/5) + target_wine_vec*(rate/5))/2
    return my_vec
