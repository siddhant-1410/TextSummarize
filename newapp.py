# Import necessary libraries
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
import numpy as np
import re

# Initialize Spacy for sentence tokenization
nlp = English()
nlp.add_pipe('sentencizer')

# Summarizer Function
def summarizer(text):
    
    doc = nlp(text.replace("\n", ""))
    sentences = [sent.text.strip() for sent in doc.sents]
    
    if len(sentences) == 0:
        return "Error: No sentences found in the input text."
    
    num_sentences = len(sentences)
    max_sent_in_summary = int(num_sentences * 0.5) ## To summarize the given text upto 50%.
    
    # Create an organizer to store sentence ordering
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
    
    # Create a TF-IDF vectorizer
    tf_idf_vectorizer = TfidfVectorizer(min_df=2, max_features=None,
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1, 3),
                                        use_idf=True,
                                        smooth_idf=True,
                                        sublinear_tf=True,
                                        stop_words='english')
    
    # Fit the vectorizer on sentences
    tf_idf_vectorizer.fit(sentences)

    sentence_vectors = tf_idf_vectorizer.transform(sentences)

    sentence_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    
   
    N = min(max_sent_in_summary, num_sentences)
    top_n_sentences = [sentences[ind] for ind in np.argsort(sentence_scores, axis=0)[::-1][:N]]
    
   
    mapped_top_n_sentences = [(sentence, sentence_organizer[sentence]) for sentence in top_n_sentences]
    
 
    mapped_top_n_sentences = sorted(mapped_top_n_sentences, key=lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in mapped_top_n_sentences]

    summary = " ".join(ordered_scored_sentences)
    
    # Remove square brackets and numbers
    summary = re.sub(r'\[\d+\]', '', summary)
    
    return summary

# Streamlit app
def main():
    st.title('Text Summarization App')
    st.subheader('Enter Text to Summarize')

    # User input for text to summarize
    text_input = st.text_area('Input Text Here:')
    
    if st.button('Summarize'):
        if text_input:
            summary = summarizer(text_input)
            st.subheader('Generated Summary:')
            st.write(summary)
        else:
            st.warning('Please enter some text to summarize.')

if __name__ == '__main__':
    main()
