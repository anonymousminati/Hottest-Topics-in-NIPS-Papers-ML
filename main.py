import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import multiprocessing

# Update preprocess_text to work with multiprocessing
PREPROCESS_REGEX = re.compile(r'[^a-zA-Z\s]')
def preprocess_text(text):
    return PREPROCESS_REGEX.sub('', text).lower()

# Add a function to preprocess text in parallel
def preprocess_texts_parallel(texts):
    with multiprocessing.Pool() as pool:
        return pool.map(preprocess_text, texts)

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

# Perform topic modeling
def perform_topic_modeling(documents, n_topics=5, n_top_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {topic_idx + 1}"] = top_features

    return topics

# Optimize plot_papers_per_year by reusing sorted data
def plot_papers_per_year(data):
    papers_per_year = data['year'].value_counts().sort_index()
    years, counts = papers_per_year.index, papers_per_year.values
    plt.figure(figsize=(10, 6))
    plt.plot(years, counts, marker='o')
    plt.title('Number of Papers Published Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.grid(True)
    plt.show()

# Update generate_word_cloud to use multiprocessing for sampling
def generate_word_cloud(text_data):
    # Limit the text data to a sample to speed up word cloud generation
    with multiprocessing.Pool() as pool:
        sample_text = ' '.join(pool.map(str, text_data[:1000]))  # Use only the first 1000 entries
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(sample_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Preprocessed Text')
    plt.show()

# Optimize analyze_future_trends by avoiding redundant filtering
def analyze_future_trends(data):
    recent_year = data['year'].max()
    recent_papers = data[data['year'] >= recent_year - 5]
    topics = perform_topic_modeling(recent_papers['processed_text'], n_topics=5, n_top_words=10)
    print("Future Trends in Machine Learning:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")

# Additional EDA steps
def display_dataset_statistics(data):
    print("Dataset Statistics:")
    print(data.describe(include='all'))
    print("\nMissing Values:")
    print(data.isnull().sum())

# Visualize the distribution of paper lengths
def plot_paper_length_distribution(data):
    data['paper_length'] = data['paper_text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    plt.hist(data['paper_length'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Paper Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Analyze the most common words
def plot_most_common_words(data, n=20):
    all_text = ' '.join(data['processed_text'])
    word_counts = Counter(all_text.split())
    most_common_words = word_counts.most_common(n)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='lightcoral')
    plt.title(f'Top {n} Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Optimize main function by reducing redundant operations
if __name__ == "__main__":
    dataset_path = "papers.csv"  # Update with the actual dataset path
    try:
        data = load_data(dataset_path)

        # Preprocess text in parallel
        data['processed_text'] = preprocess_texts_parallel(data['paper_text'])

        # Display dataset statistics
        display_dataset_statistics(data)

        # Plot papers per year
        plot_papers_per_year(data)

        # Plot paper length distribution
        plot_paper_length_distribution(data)

        # Generate word cloud
        generate_word_cloud(data['processed_text'])

        # Plot most common words
        plot_most_common_words(data)

        # Analyze future trends
        analyze_future_trends(data)

        # Perform LDA analysis
        topics = perform_topic_modeling(data['processed_text'], n_topics=5, n_top_words=10)
        for topic, words in topics.items():
            print(f"{topic}: {', '.join(words)}")
    except Exception as e:
        print(f"Error: {e}")