Overview

The HR Response Generator is a web-based application designed to automate and enhance human resources communication. It leverages advanced natural language processing techniques, including OpenAI's GPT-3.5 model, to generate contextually relevant responses to HR-related queries. The application is built with Streamlit, making it highly interactive and user-friendly.

Uses FAISS and OpenAI embeddings for data vectorisation , scikit learn for a cosine similarity calculation and GPT 3.5T for response generation. Deployed on Streamlit Web.

Installation

-Populate the training data.csv with email/message and answer pairs that you want to base responses on or alternatively  sections of your contract/award.

-insert your OpenAI API key on the .env file

-modify the "template" in the LLMchain section of the code
