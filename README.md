# Natural Language Processing (NLP) Project

This repository contains a comprehensive collection of Jupyter notebooks demonstrating various Natural Language Processing (NLP) techniques and concepts using the Natural Language Toolkit (NLTK). The project serves as both a learning resource and a practical implementation guide for fundamental NLP tasks.

## 📚 Project Contents

The project includes the following notebooks, each focusing on a specific NLP concept:

1. **Tokenization** (`Tokenization.ipynb`)
   - Text tokenization techniques using NLTK
   - Word and sentence tokenization
   - Implementation of different tokenizers (word_tokenize, wordpunct_tokenize, TreebankWordTokenizer)

2. **Stopwords** (`Stopwords.ipynb`)
   - Understanding and removing stopwords using NLTK
   - Working with stopwords in different languages
   - Custom stopword list implementation

3. **Stemming** (`Stemming.ipynb`)
   - Porter Stemmer implementation
   - RegexpStemmer for custom stemming rules
   - SnowballStemmer implementation
   - Comparison of different stemming approaches

4. **Lemmatization** (`Lemmatization.ipynb`)
   - WordNet Lemmatizer implementation
   - Comparison with stemming techniques
   - Practical applications

5. **Parts of Speech Tagging** (`Parts Of Speech Tagging.ipynb`)
   - POS tagging using NLTK's averaged_perceptron_tagger
   - Custom tagger implementation
   - Practical examples and applications

6. **Bag of Words** (`Bag Of Words.ipynb`)
   - Text vectorization using NLTK
   - Count vectorization implementation
   - Text preprocessing and feature extraction

7. **BOW and N-Gram** (`BOW and N-Gram.ipynb`)
   - N-gram models implementation
   - Combining with Bag of Words
   - Feature extraction techniques

8. **TF-IDF** (`TF-IDF.ipynb`)
   - Term Frequency-Inverse Document Frequency implementation
   - Text preprocessing and feature extraction
   - Practical applications

9. **Word2Vec** (`Word2Vec.ipynb`)
   - Word embeddings implementation
   - CBOW and Skip-gram models
   - Word vector visualization and analysis

## 📁 Dataset

The project includes a sample dataset:
- `spam.csv`: A dataset for spam detection, containing 5573 entries of SMS messages labeled as spam or ham (legitimate)

## 🛠️ Requirements

To run the notebooks in this project, you'll need the following Python packages:
- numpy
- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using pip:
```bash
pip install numpy pandas nltk scikit-learn matplotlib seaborn
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 📝 Usage

Each notebook is self-contained and includes:
- Theoretical explanations
- Code implementations using NLTK
- Examples and visualizations
- Practice exercises

The notebooks are designed to be followed in sequence, starting from basic concepts (Tokenization, Stopwords) to more advanced topics (Word2Vec, TF-IDF).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the terms of the license included in the repository. See the `LICENSE` file for more details.

## 📧 Contact

For any questions or suggestions, please open an issue in the repository.

---
*Note: This project is intended for educational purposes and serves as a comprehensive guide to NLP concepts and implementations using NLTK.*