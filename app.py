from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import json
from dotenv import load_dotenv
import os 

app = Flask(__name__)




class Mentor:
    def __init__(self, Name, Region, Languages, Subject, Preferred_Time, Preferred_Days, Mode, ExperiencePhysics, ExperienceChemistry, ExperienceMaths, Score):
        self.Name = Name
        self.Region = Region
        self.Languages = Languages
        self.Subject = Subject
        self.Preferred_Time = Preferred_Time
        self.Preferred_Days = Preferred_Days
        self.Mode = Mode
        self.ExperiencePhysics = ExperiencePhysics
        self.ExperienceChemistry = ExperienceChemistry
        self.ExperienceMaths = ExperienceMaths
        self.Score = Score



#loading the dataset
df = pd.read_csv('./dataset.csv')




@app.route('/get_matches', methods=['POST'])
def get_matches():
    query = request.get_json()
    f = 30  # Length of the vectors you are indexing
    t = AnnoyIndex(f, 'angular')
    t.load('test.ann')  # super fast, will just mmap the file
    loaded_pca = pickle.load(open('pca_transformer.pkl', 'rb'))
    loaded_vectorizer = pickle.load(open('tfidf_transformer.pkl', 'rb'))
    loaded_encoder = pickle.load(open('onehot_transformer.pkl', 'rb'))
    loaded_scaler = pickle.load(open('ss_transformer.pkl', 'rb'))
    query_preferred_time_features = loaded_vectorizer.transform([query['Preferred_Time']]).toarray()

    query_other_features = loaded_encoder.transform([[query['Region'], query['Languages'], query['Mode']]])
    query_features = np.concatenate([query_other_features, [[query['ExperiencePhysics'], query['ExperienceChemistry'], query['ExperienceMaths']]], query_preferred_time_features], axis=1)
    query_features = query_features * 1 #manual weights (IMP -> dont forget to assign similar weights of pre training)
    query_features = loaded_scaler.transform(query_features)
    query_features = loaded_pca.transform(query_features)

    indices = t.get_nns_by_vector(query_features[0], 10)
    scores = [1 - t.get_distance(i, indices[0]) for i in indices]

    # Sort teachers by similarity scores in descending order
    sorted_indices = [i for _, i in sorted(zip(scores, indices), reverse=True)]
    sorted_scores = sorted(scores, reverse=True)
    sorted_scores , sorted_indices = sorted_scores[:10] , sorted_indices[:10]

    temp = df.iloc[sorted_indices]
    temp.loc[: , ["Score"]]= sorted_scores

    mentors = [Mentor(row['Name'], row['Region'], row['Languages'], row['Subject'], row['Preferred_Time'], row['Preferred_Days'], row['Mode'], row['ExperiencePhysics'], row['ExperienceChemistry'], row['ExperienceMaths'], row['Score']) for index, row in temp.iterrows()]
    mentors_dict = [mentor.__dict__ for mentor in mentors]
    # Convert the list of dictionaries to a JSON string
    mentors_json = json.dumps(mentors_dict)
    mentors_json

    return jsonify(mentors_json)

if __name__ == '__main__':
    app.run(debug=True)