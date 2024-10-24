import re
from nltk.tokenize import word_tokenize
import pymorphy3
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import tqdm
import pandas as pd
import numpy as np


def preprocess_data(df, stopwords=None, clean=True):
    assert not (clean and stopwords is None), 'Stop-words should be specified when clean=True'
    morph = pymorphy3.MorphAnalyzer()
    new_df = df.copy()
    
    if clean:
        new_df['text'] = new_df['text'].apply(lambda text: re.sub('[?!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”)*°(°*``,.]', '', text))

    new_df['text'] = new_df['text'].apply(lambda text: word_tokenize(text.lower()))
    new_df['text'] = new_df['text'].apply(lambda text: [morph.parse(word)[0].normal_form for word in text if not clean or word not in stopwords])
    new_df['text'] = new_df['text'].apply(lambda text: ' '.join(text))
    return new_df

def docs_to_vectors(data: pd.DataFrame, vectorizer: KeyedVectors):
    zero_samples = []
    zero_samples_count = 0
    text_data = data['text'].map(word_tokenize)
    all_vector_representation = []
    for document in tqdm.tqdm(text_data):
        doc_words = [word for word in document if word in vectorizer.key_to_index]
        if doc_words:
            vector_representation = sum([np.array(vectorizer[word]) for word in doc_words])
            vector_representation /= np.linalg.norm(vector_representation, ord=2)
        else:
            vector_representation = np.zeros(300,)
            zero_samples_count += 1
            zero_samples.append(document)

        all_vector_representation.append(vector_representation)

    all_vector_representation = np.stack(all_vector_representation, axis=0)
    print('Количество примеров, не превращённых в векторы (нулевых векторов):', zero_samples_count)
    return all_vector_representation, zero_samples