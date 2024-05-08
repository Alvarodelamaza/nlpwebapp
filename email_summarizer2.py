import streamlit as st
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


# create mesages for the completions.create method of OpenAI completions





def summarize_email(text, summary_length):
    # Load the pre-trained spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process the email text with spaCy
    doc = nlp(text)

    # Tokenize the document and count the word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    # Normalize the frequencies by the highest frequency
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Score sentences based on the word frequencies
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text]
                else:
                    sentence_scores[sent] += word_frequencies[word.text]

    # Select the top n sentences with the highest scores
    summarized_sentences = nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    final_summary = ' '.join([sent.text for sent in summarized_sentences])

    return final_summary



# Setting up the title of the app
st.title(" 	:page_facing_up: Text Summarizer")

# Text area for user input
user_input = st.text_area(" :pencil: Enter the text you want to summarize", height=150)

length = st.number_input("Select summary size:", min_value=0, max_value=10, step=1, value=3)

# Button to trigger summarization
if st.button("Summarize"):
    if user_input:
        try:
            summary = summarize_email(user_input,length)
            st.write(summary)
        except:
            st.write("Your text is smaller than the selected length, please select an smaller size")
    else:
        st.write("Please enter some text to summarize.")

st.caption("This NLP tool has been designed for education purposes for the Vrije Universiteit Amsterdam by Jakob Walter, Doruk Tarhan and Alvaro de la Maza . The  model uses spaCy : Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing")