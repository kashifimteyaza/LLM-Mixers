#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle


# In[4]:


import os
print(os.getcwd())


# In[5]:


# Get the path to the user's home directory
home_dir = os.path.expanduser("~")

# Construct the path to the Downloads folder
downloads_path = os.path.join(home_dir, "Downloads")

# Construct the full path to the CSV file
file_path = os.path.join(downloads_path, "netflix_titles.csv")


# In[6]:


df = pd.read_csv(file_path)


# In[7]:


print (df.head())


# In[8]:


df


# In[9]:


def create_textual_representation(row):
    textual_representation = f"""Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},
Description: {row['description']}"""
    return textual_representation


# In[10]:


df['textual_representation'] = df.apply(create_textual_representation, axis=1)


# In[11]:


df


# In[12]:


print (df['textual_representation'].values[1])


# In[13]:


import faiss


# In[14]:


import requests
import numpy as np

dim = 4096

index = faiss.IndexFlatL2(dim)

X = np.zeros((len(df['textual_representation']), dim), dtype= 'float32')


# In[15]:


X


# In[16]:


def get_embedding(text):
    res = requests.post('http://localhost:11434/api/embeddings',
                        json={
                            'model': 'llama2',
                            'prompt': text
                        })
    return res.json()['embedding']

# Function to process a batch of items
def process_batch(batch):
    return [get_embedding(text) for text in batch]

# Check if cached embeddings exist
cache_file = 'embeddings_cache.pkl'
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        X = pickle.load(f)
    print("Loaded embeddings from cache.")
else:
    # Batch size and number of workers
    batch_size = 10
    num_workers = 5

    # Create batches
    batches = [df['textual_representation'][i:i+batch_size] for i in range(0, len(df), batch_size)]

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        # Use tqdm for a progress bar
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Generating Embeddings")):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            X[start_idx:end_idx] = future.result()

    # Cache the embeddings
    with open(cache_file, 'wb') as f:
        pickle.dump(X, f)
    print("Saved embeddings to cache.")

# Add vectors to the index
index.add(X)


# In[51]:


index= faiss.write_index(index, 'index')


# In[69]:


df [df.title.str.contains ('Unbroken')]


# In[70]:


favorite_movie = df.iloc[3595]


# In[71]:


res = requests.post('http://localhost:11434/api/embeddings',
                        json={
                            'model': 'llama2',
                            'prompt': favorite_movie ['textual_representation']
                        })


# In[75]:


embedding = np.array([res.json()['embedding']], dtype = 'float32')

D, I = index.search(embedding, 5)


# In[76]:


best_matches = np.array(df['textual_representation']) [I.flatten()]


# In[64]:


for match in best_matches :
    print('Next Movie')
    print(match)
    print()


# 

# In[ ]:





# In[ ]:




