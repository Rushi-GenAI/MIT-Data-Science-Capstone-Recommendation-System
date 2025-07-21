Music Recommendation System
Project Overview
This project, developed as a capstone for my MIT Data Science certification, focuses on building a robust music recommendation system. In today's digital music landscape, keeping users engaged is paramount for platform growth and revenue. This system aims to enhance user engagement by recommending new songs they are likely to enjoy, thereby providing added value beyond their usual favorites.

The core objective is to answer key questions:

Can we recommend the top 10 songs for users based on their listening habits?

Can we achieve a reliable recommendation system that users will anticipate and appreciate?

Data science techniques are employed to understand existing data, extract relevant insights, and build various models to identify the best possible solution for personalized music recommendations.

Problem Formulation
The project addresses the challenge of recommending new songs to users that align with their preferences, thereby increasing user engagement and satisfaction on music platforms. This is achieved by analyzing historical user-song interaction data and leveraging machine learning models to predict user preferences for unlistened songs.

Data Source and Dictionary
The system utilizes the Taste Profile Subset released by the Echo Nest as part of the Million Song Dataset. This dataset comprises two main files:

song_data.csv: Contains details about individual songs.

song_id: A unique identifier for each song.

title: The title of the song.

release: The name of the album the song was released on.

artist_name: The name of the artist.

year: The year of the song's release.

count_data.csv: Contains user listening activity.

user_id: A unique identifier for each user.

song_id: The unique identifier for the song.

play_count: The number of times the user played the song.

Data Source Link: http://millionsongdataset.com/

Technical Stack and Libraries
Programming Language: Python

Data Manipulation: pandas, numpy

Data Visualization: matplotlib.pyplot, seaborn

Machine Learning / Recommendation Libraries:

sklearn.metrics.pairwise (for cosine_similarity)

collections (for defaultdict)

sklearn.metrics (for mean_squared_error)

surprise (for collaborative filtering models like SVD, KNNBasic)

imblearn.over_sampling (for SMOTE - used for handling imbalanced data)

nltk (for text preprocessing: punkt, stopwords, wordnet)

re (for regular expressions)

sklearn.feature_extraction.text (for TfidfVectorizer)

Project Structure
Rushikesh_Chougule_Music_Recommendation_System_Full_Code_Capstone.ipynb: The main Jupyter Notebook containing all the code for data loading, preprocessing, EDA, model building, evaluation, and recommendation generation.

count_data.csv: Dataset containing user play counts for songs.

song_data.csv: Dataset containing song metadata.

df_final.csv (Optional, if saved during execution): Processed and filtered dataset.

Project Presentation.pdf (Placeholder): Your presentation explaining the project.

Key Steps and Methodology
Data Loading and Initial Exploration:

Loading count_data.csv and song_data.csv.

Initial inspection of data types, missing values, and data distribution.

Handling missing values by filling NaN in 'release' and 'title' columns.

Merging count_df and song_df to create a unified dataset (df).

Data Preprocessing:

Label Encoding: Encrypting user_id and song_id into numeric features using LabelEncoder.

Data Filtering (Cold Start Problem Mitigation):

Filtering out users who have listened to fewer than 90 songs (NSONGS_CUTOFF = 90).

Filtering out songs listened to by fewer than 120 users (NSONG_CUTOFF = 120).

Further filtering to include only songs with play_count less than or equal to 5, as these constitute a large portion of the data.

Exploratory Data Analysis (EDA):

Analysis of play_count distribution to understand user engagement patterns.

Recommendation System Approaches:

Popularity-Based Recommendation System:

Calculated average play count and frequency of each song.

Developed a function to recommend top N songs based on popularity, with a minimum play count threshold.

Observation: Top songs should be easily accessible to users (e.g., as a portlet on the music platform).

User-User Similarity-Based Collaborative Filtering:

Utilized the surprise library for building collaborative filtering models.

Trained and evaluated KNNBasic models with different similarity metrics (Cosine, MSD, Pearson).

Addressing Data Imbalance with SMOTE: Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to balance the play_count distribution, improving model performance, especially for less frequent interactions.

Evaluated model performance using RMSE, MAE, FCP, Precision, and Recall.

Developed a function to generate top N recommendations for a specific user based on user-user similarity.

Item-Item Similarity-Based Collaborative Filtering:

Trained and evaluated KNNBasic models with different similarity metrics (Cosine, MSD, Pearson) based on item-item similarity.

Evaluated model performance using RMSE, MAE, FCP, Precision, and Recall.

Developed a function to generate top N recommendations for a specific user based on item-item similarity.

Model-Based Collaborative Filtering - Matrix Factorization (SVD):

Implemented Singular Value Decomposition (SVD) using the surprise library.

Tuned hyperparameters using GridSearchCV for optimal performance.

Applied SVD to the oversampled dataset.

Evaluated the optimized SVD model.

Developed a function to generate top N recommendations for a specific user using the SVD algorithm.

Content-Based Recommendation System (Text-Based):

Created a combined text feature from title, release, and artist_name.

Text Preprocessing: Implemented a tokenize function for:

Lowercasing and removing non-alphabetical characters.

Tokenization using nltk.word_tokenize.

Removing English stopwords.

Lemmatization using WordNetLemmatizer.

TF-IDF Vectorization: Used TfidfVectorizer to convert text data into numerical vectors.

Cosine Similarity: Computed cosine similarity between song TF-IDF vectors to find similar songs.

Developed a function to recommend similar songs based on a given song title.

Results and Findings
The dataset was successfully cleaned and preprocessed, including handling missing values and encoding categorical IDs.

Data filtering based on user and song interaction counts significantly improved computational efficiency and model relevance.

Popularity-based recommendations serve as a good baseline and can be used for new users or cold-start scenarios.

Collaborative filtering models (User-User, Item-Item, and SVD) were implemented and evaluated.

Oversampling with SMOTE proved beneficial in addressing the class imbalance in play counts, leading to improved recall.

SVD, especially with optimization and oversampling, provided acceptable results for predicting play counts for unknown user-song interactions.

How to Run the Project
Clone the Repository:

git clone https://github.com/your-username/MIT-Data-Science-Capstone-Recommendation-System.git
cd MIT-Data-Science-Capstone-Recommendation-System

Download Data:

Download count_data.csv and song_data.csv from http://millionsongdataset.com/ (specifically the Taste Profile Subset).

Place these files in the same directory as the Jupyter Notebook, or update the file paths in the notebook accordingly.

Set up Environment:

It's recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install Dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise imblearn nltk

After installing nltk, run the following commands in a Python interpreter or a new cell in your Jupyter Notebook to download necessary NLTK data:

import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

Run the Jupyter Notebook:

jupyter notebook Rushikesh_Chougule_Music_Recommendation_System_Full_Code_Capstone.ipynb

Open the notebook in your browser and run all cells sequentially to reproduce the analysis and results.

Contact
Your Name: Rushikesh Chougule

LinkedIn: www.linkedin.com/in/rushikesh-kiran-chougule
