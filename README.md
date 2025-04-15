# Hottest Topics in Machine Learning

This project explores the hottest topics in machine learning using NIPS (Neural Information Processing Systems) papers. It involves preprocessing text data, performing exploratory data analysis (EDA), and applying Latent Dirichlet Allocation (LDA) for topic modeling.

## Project Structure

```
Hottest_Topic_Modeling.ipynb  # Jupyter Notebook with step-by-step implementation
kaggle.py                     # Script for Kaggle API integration (if applicable)
main.py                       # Main Python script for running the project
papers.csv                    # Dataset containing NIPS papers
requirements.txt              # List of required Python packages
```

## Steps in the Project

### 1. Loading the Dataset
The dataset (`papers.csv`) contains NIPS papers with the following columns:
- `id`: Unique identifier for each paper
- `year`: Year of publication
- `title`: Title of the paper
- `event_type`: Type of event (e.g., conference)
- `pdf_name`: Name of the PDF file
- `abstract`: Abstract of the paper
- `paper_text`: Full text of the paper

### 2. Preprocessing the Text Data
- Remove special characters and numbers.
- Convert text to lowercase.
- Use multiprocessing for faster preprocessing.

### 3. Exploratory Data Analysis (EDA)
- **Number of Papers Per Year**: Visualize the number of papers published each year.
- **Paper Length Distribution**: Analyze the distribution of paper lengths.
- **Word Cloud**: Generate a word cloud to visualize the most common words.
- **Most Common Words**: Plot the top 20 most common words in the dataset.

### 4. Topic Modeling with LDA
- Use `TfidfVectorizer` for efficient vectorization.
- Apply `LatentDirichletAllocation` to identify topics.
- Optimize LDA with fewer iterations and batch learning for faster computation.

### 5. Analyzing Trends
- Analyze trends in machine learning by examining topics from recent papers.

## How to Run the Project

### Prerequisites
- Python 3.8 or later
- Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Jupyter Notebook
1. Open `Hottest_Topic_Modeling.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the steps in the notebook to execute the project.

### Running the Python Script
1. Ensure the dataset (`papers.csv`) is in the same directory as `main.py`.
2. Run the script:
   ```bash
   python main.py
   ```

## Results
- **Topics Identified**: The LDA model identifies the most representative topics in the dataset.
- **Visualizations**: Includes plots for EDA and a word cloud.
- **Future Trends**: Insights into recent trends in machine learning.

## Dataset
The dataset used in this project is publicly available and contains NIPS papers. Ensure the dataset is placed in the root directory of the project.

## Contributions
Feel free to contribute to this project by improving the code, adding new features, or optimizing the existing implementation.

