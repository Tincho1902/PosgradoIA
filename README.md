# NLP - PosgradoIA_UBA

AI-Specialization - UBA (Universidad Nacional de Buenos Aires) - NLP Projects


## TextVectorization

![img1](images/1.png)
Link to colab: **[TextVectorization](https://github.com/Tincho1902/PosgradoIA/blob/main/1a_vectorizacion.ipynb)**

Text vectorization is the process of converting text data into numerical vectors that can be used for machine learning or other applications. There are different methods of text vectorization, such as:

- **Count-based methods**: These methods use the frequency or occurrence of words in a text to create vectors. For example, a bag-of-words (BoW) vector has the length of the entire vocabulary and the values represent how many times each word appears in a text. Another example is TF-IDF, which assigns higher weights to words that are more distinctive or informative in a text.
- **Embedding methods**: These methods use neural networks or other algorithms to learn the semantic and syntactic relationships between words and create vectors that capture these meanings. For example, Word2Vec creates word vectors based on the context of words in a text. Another example is BERT, which creates word and sentence vectors based on the bidirectional context and attention mechanism.

Origin:

(1) tf.keras.layers.TextVectorization | TensorFlow v2.14.0. 

https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization.

(2) What Is Text Vectorization? Everything You Need to Know - deepset. 

https://www.deepset.ai/blog/what-is-text-vectorization-in-nlp.

(3) Convert SVG text to vector graphics online for free - Aspose. 

https://products.aspose.app/svg/text-to-vector.

(4) Text Vectorization and Word Embedding | Guide to Master NLP (Part 5). 

https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/.

## Information Retrieval System with NLTK

![img1](images/2.png)
Link to colab: **[Information Retrieval System](https://github.com/Tincho1902/PosgradoIA/blob/main/2c%20-%20bot_tfidf_nltk.ipynb)**

An information retrieval system is a software that can process a large collection of documents and find the ones that are relevant to a user's query. Using NLTK, one can build an information retrieval system by following these steps:

- Preprocess the documents and the query by tokenizing, normalizing, and removing stopwords.
- Represent the documents and the query as vectors of term weights, using methods such as TF-IDF, LDA, LSI, or log entropy.
- Compute the similarity between the query vector and each document vector, using measures such as cosine similarity or Jaccard coefficient.
- Rank the documents by their similarity scores and return the top-k results to the user.

Origin:

(1) 7 Extracting Information from Text - NLTK. 

https://www.nltk.org/book_1ed/ch07.html.

(2) yolanda93/information_retrieval_system - GitHub. 

https://github.com/yolanda93/information_retrieval_system.

(3) nionios / Information Retrieval with NLTK · GitLab.

https://gitlab.com/nionios/information-recovery-nltk.

(4) NLTK :: Natural Language Toolkit. 

https://www.nltk.org/.

## Custom embedddings with Gensim

![img1](images/3.png)
Link to colab: **[Custom embedddings](https://github.com/Tincho1902/PosgradoIA/blob/main/3b_Custom_embedding_con_Gensim.ipynb)**

Custom embeddings with Gensim are word embeddings that can be train on text data using Python library. Gensim is an open source library that provides various tools and resources for natural language processing, such as topic modeling, text summarization, and word embedding. 

To create custom embeddings with Gensim:

- **Preprocess text data**: You need to clean, tokenize, normalize, and stem your text data to remove noise and reduce the vocabulary size.
- **Choose a word embedding model**: You need to select a word embedding model that suits your task and data, such as Word2Vec, FastText, or GloVe. Gensim provides implementations of these models that you can use or modify.
- **Train word embedding model**: You need to train your word embedding model on your text data using Gensim's methods and classes. You can also tune the hyperparameters, such as the vector size, the window size, the learning rate, etc.
- **Save and load word embedding model**: You need to save your word embedding model in a file format that you can later load and use for other applications. Gensim supports various file formats, such as binary, text, or pickle.

Origin:

(1) NLP - Custom word-embeddings in gensim - Stack Overflow. 

https://stackoverflow.com/questions/72108143/custom-word-embeddings-in-gensim.

(2) How to Develop Word Embeddings in Python with Gensim.

https://machinelearningmastery.com/develop-word-embeddings-python-gensim/.

(3) Custom Gensim Embeddings in Rasa | The Rasa Blog | Rasa. 

https://rasa.com/blog/custom-gensim-embeddings-in-rasa/.

## Next word prediction

![img1](images/4.png)
Link to colab: **[Next word prediction](https://github.com/Tincho1902/PosgradoIA/blob/main/4d%20-%20predicci%C3%B3n_palabra_tensorflow.ipynb)**

Next word prediction models are models that can generate the most likely word to follow a given sequence of words. They are useful for applications such as text completion, text generation, machine translation, and speech recognition. Next word prediction models are based on natural language processing (NLP), which is the field of computer science that deals with understanding and manipulating natural language.

There are different types of next word prediction models, such as:

- **N-gram models**: These models use the frequency of n consecutive words in a large corpus of text to estimate the probability of the next word. For example, a bigram model uses the previous word, and a trigram model uses the previous two words, to predict the next word. N-gram models are simple and fast, but they have limitations such as data sparsity and lack of context.
- **Neural network models**: These models use artificial neural networks, which are computational systems that can learn from data, to predict the next word. Neural network models can capture complex patterns and dependencies in language, and can generate more diverse and fluent texts. Some examples of neural network models are recurrent neural networks (RNNs), long short-term memory networks (LSTMs), and transformers.

If you want to learn more about next word prediction models, you can check out these resources:

- [Next Word Prediction Model using Python](https://thecleverprogrammer.com/2023/07/17/next-word-prediction-model-using-python/): This is a tutorial that shows how to build a next word prediction model using Python and LSTM.
- [Next Word Prediction with Deep Learning Models](https://link.springer.com/chapter/10.1007/978-3-031-09753-9_38): This is a research paper that compares different deep learning models for next word prediction on Turkish corpus.
- [Exploring the Next Word Predictor!](https://link.springer.com/chapter/10.1007/978-3-031-09753-9_38): This is a blog post that explains the concepts and implementation of n-gram and LSTM models for next word prediction.
- [Stacked Language Models for an Optimized Next Word Generation](https://towardsdatascience.com/exploring-the-next-word-predictor-5e22aeb85d8f): This is another research paper that proposes a stacked language model that combines three models for next word generation.
- [Next Word Prediction Using Deep Learning: A Comparative Study](https://ieeexplore.ieee.org/document/9845545): This is yet another research paper that evaluates different deep learning models for next word prediction on English corpus.

Origin:

(1) next-word-prediction · GitHub Topics · GitHub. 

https://github.com/topics/next-word-prediction.

(2) Next Word Prediction with Deep Learning Models | SpringerLink. 

https://link.springer.com/chapter/10.1007/978-3-031-09753-9_38.

(3) Next Word Prediction Model using Python | Aman Kharwal. 

https://thecleverprogrammer.com/2023/07/17/next-word-prediction-model-using-python/.

(4) Next Word Prediction with Deep Learning Models | SpringerLink. 

https://link.springer.com/chapter/10.1007/978-3-031-09753-9_38.

(5) Exploring the Next Word Predictor! - Towards Data Science. 

https://towardsdatascience.com/exploring-the-next-word-predictor-5e22aeb85d8f.

(6) Stacked Language Models for an Optimized Next Word Generation. 

https://ieeexplore.ieee.org/document/9845545.


## Sentiment analysis with Embeddings + LSTM

![img1](images/5.png)
Link to colab: **[Sentiment analysis](https://github.com/Tincho1902/PosgradoIA/blob/main/5-clothing-ecommerce-reviews.ipynb)**

Sentiment analysis is a technique that uses natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information in a text. It can help to understand the opinions, emotions, and intentions of customers, users, or other stakeholders in various domains and channels.

Some examples of applications of sentiment analysis are:

- **Customer feedback analysis**: Sentiment analysis can help businesses to understand the satisfaction and loyalty of their customers by analyzing their reviews, ratings, surveys, comments, etc. This can help to improve customer service, product quality, marketing strategies, etc.
- **Social media analysis**: Sentiment analysis can help to monitor the public opinion and trends on social media platforms, such as Twitter, Facebook, Instagram, etc. This can help to identify the influencers, the sentiments, the topics, the hashtags, etc., related to a brand, a product, a service, or a person.
- **Market research**: Sentiment analysis can help to conduct market research and analysis by collecting and analyzing data from various sources, such as news articles, blogs, forums, reports, etc. This can help to understand the market needs, demands, opportunities, threats, etc.
- **Healthcare**: Sentiment analysis can help to improve healthcare services and outcomes by analyzing the emotions and sentiments of patients and caregivers from various sources, such as medical records, clinical notes, online forums, etc. This can help to diagnose mental health conditions, monitor patient satisfaction and well-being, provide personalized care and support, etc.

Origin:

(1) Sentiment analysis - Wikipedia. 

https://en.wikipedia.org/wiki/Sentiment_analysis.

(2) What is Sentiment Analysis? A Complete Guide for Beginners. 

https://www.freecodecamp.org/news/what-is-sentiment-analysis-a-complete-guide-to-for-beginners/.

(3) ¿Qué es el análisis de sentimiento? | Microsoft Dynamics 365. 

https://dynamics.microsoft.com/es-es/ai/customer-insights/what-is-sentiment-analysis/.

## Question-Answering ChatBOT with LSTM

![img1](images/6.png)
Link to colab: **[ChatBOT with LSTM](https://github.com/Tincho1902/PosgradoIA/blob/main/6-bot-qa.ipynb)**

A chatbot to question-answering using LSTM is a chatbot that uses LSTM to encode the user's question and decode the answer from the relevant data. LSTM can help the chatbot to understand the meaning and context of the question, and generate a coherent and accurate answer. Some examples of chatbot to question-answering using LSTM are:

- [Neural-chatbot](^1^): This is a GitHub repository that shows how to build a question-answer style chatbot using LSTM for language models with attention mechanism. The chatbot uses Twitter and Cornell movie subtitle datasets as the data source.
- [Conversational_Chatbot_using_LSTM](^2^): This is another GitHub repository that shows how to create a conversational chatbot using sequence to sequence LSTM models. The chatbot uses Chatterbot Kaggle English dataset as the data source.
- [Chatbot-Using-LSTM](^3^): This is yet another GitHub repository that shows how to implement a retrieval-based chatbot for a ticketing portal using TensorFlow and LSTM. The chatbot uses a custom dataset of queries and responses as the data source.
- [Build a generative chatbot using recurrent neural networks (LSTM RNNs)](^4^): This is a tutorial that shows how to build a generative chatbot using LSTM RNNs and Keras. The chatbot uses a custom dataset of jokes as the data source.

Origin:

(1) GitHub - liambll/neural-chatbot: Chatbot with LSTM Sequence To Sequence .... 

https://github.com/liambll/neural-chatbot.

(2) ShrishtiHore/Conversational_Chatbot_using_LSTM - GitHub. 

https://github.com/ShrishtiHore/Conversational_Chatbot_using_LSTM.

(3) Chatbot-Using-LSTM - GitHub. 

https://github.com/dasnikita/Chatbot-Using-LSTM.

(4) Build a generative chatbot using recurrent neural networks (LSTM RNNs .... 

https://hub.packtpub.com/build-generative-chatbot-using-recurrent-neural-networks-lstm-rnns/.
