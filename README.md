```markdown
# 🎵 Music Recommendation System

## 📌 Project Overview

This capstone project, developed for the **MIT Data Science Certification**, focuses on building a robust **music recommendation system** that enhances user engagement by suggesting new songs users are likely to enjoy.

With the massive scale of digital music platforms, delivering personalized recommendations is crucial for retaining users and increasing satisfaction. This system addresses key questions:

- Can we recommend the **top 10 songs** for users based on their listening habits?
- Can we build a **reliable and engaging** recommendation experience?

By leveraging **data science** and **machine learning**, this project analyzes user behavior and builds models for personalized music suggestions.

---

## 🎯 Problem Formulation

The challenge is to recommend songs aligned with individual user preferences, using **historical user-song interaction data**. By applying advanced ML techniques, the system predicts which new songs users are most likely to enjoy—ultimately enhancing user experience and platform retention.

---

## 📊 Data Source and Dictionary

**Dataset:** [Million Song Dataset - Taste Profile Subset](http://millionsongdataset.com/)

### Files Used:
- **`song_data.csv`** — Song metadata:
  - `song_id`, `title`, `release`, `artist_name`, `year`
- **`count_data.csv`** — User interaction data:
  - `user_id`, `song_id`, `play_count`

---

## 🧰 Tech Stack and Libraries

- **Language**: Python  
- **Data Manipulation**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **ML / Recommendation**:
  - `scikit-learn`, `surprise`, `imblearn`
  - `nltk` (for text preprocessing)
  - `re`, `collections`, `TfidfVectorizer`

---

## 🗂️ Project Structure

```

MIT-Data-Science-Capstone-Recommendation-System/
├── Rushikesh\_Chougule\_Music\_Recommendation\_System\_Full\_Code\_Capstone.ipynb
├── song\_data.csv
├── count\_data.csv
├── df\_final.csv (optional, generated)
└── Project Presentation.pdf

````

---

## 🔍 Key Steps and Methodology

### 1. Data Loading & Cleaning
- Loaded and merged `count_data.csv` with `song_data.csv`.
- Handled missing values in song metadata.

### 2. Preprocessing
- Label encoded `user_id` and `song_id`.
- Filtered:
  - Users with < 90 songs.
  - Songs with < 120 listeners.
  - Interactions with `play_count` > 5.

### 3. Exploratory Data Analysis (EDA)
- Analyzed `play_count` distribution and user behavior trends.

### 4. Recommendation Systems Implemented

#### 🔹 Popularity-Based
- Recommended most-played songs.
- Acts as a baseline or cold-start fallback.

#### 🔹 Collaborative Filtering

- **User-User Similarity (KNNBasic)**
- **Item-Item Similarity (KNNBasic)**
- **Matrix Factorization (SVD)**
  - Tuned using `GridSearchCV`.
  - Performed well on sparse data.

- **SMOTE** was applied to handle play count imbalance.

- Metrics: **RMSE**, **MAE**, **FCP**, **Precision**, **Recall**

#### 🔹 Content-Based (Text)
- Created text from `title`, `release`, and `artist_name`.
- Preprocessed using:
  - Tokenization
  - Stopword removal
  - Lemmatization
- Used **TF-IDF + Cosine Similarity** to recommend similar songs.

---

## 🏁 Results & Takeaways

- **Data Filtering** significantly improved performance and reduced noise.
- **Popularity-based** approach works well for new users.
- **Collaborative filtering** (especially **SVD with SMOTE**) gave the most balanced performance.
- **Content-based** recommendations enhance diversity and novelty in recommendations.

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MIT-Data-Science-Capstone-Recommendation-System.git
cd MIT-Data-Science-Capstone-Recommendation-System
````

### 2. Download the Dataset

Download `count_data.csv` and `song_data.csv` from:
👉 [Million Song Dataset](http://millionsongdataset.com/)

Place them in the same directory as the notebook.

### 3. Set up the Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise imblearn nltk
```

Then run in Python:

```python
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Run the Notebook

```bash
jupyter notebook Rushikesh_Chougule_Music_Recommendation_System_Full_Code_Capstone.ipynb
```

---

## 👤 Contact

**Name**: Rushikesh Chougule
**LinkedIn**: [linkedin.com/in/rushikesh-kiran-chougule](https://www.linkedin.com/in/rushikesh-kiran-chougule)
