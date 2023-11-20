#!/opt/homebrew/bin/python3


import streamlit as st
import numpy as np
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity



load_dotenv()
# 1. Vectorise the sales responcse csv data
loader = CSVLoader(file_path="./HR Training dataset .csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search
def retrieve_info(query):
    similar_responses = db.similarity_search(query, k=1)
    best_practice = [doc.page_content for doc in similar_responses]
    
    return best_practice

# Calculate similarity scores between message and output of similarity search function
def get_similarity(message, best_practice):
    query_embedding = np.array(embeddings.embed_query(message)).reshape(1, -1)
    closest_match_embedding = np.array(embeddings.embed_query(best_practice[0])).reshape(1, -1)

    # Calculate cosine similarity 
    similarity = cosine_similarity(query_embedding, closest_match_embedding)[0][0]

    return similarity

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

template = """
You are a [insert title] at the [industry and company].
Your company offers [service/product] and you are [job function/task].
I will share a candidate's message with you and you will give me the best answer that 
I should send to this candidate based on past responses.

Below is a message I received from the candidate:
{message}

Here is a list of best practies of how we normally respond to candidate in similar scenarios:
{best_practice}

Please write the best response that I should send to this candidate:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation

def generate_response(message):
    best_practice = retrieve_info(message)
    print(best_practice)
    similarity = get_similarity(message, best_practice)
    print(similarity)
    # Reject the match if it is not similar enough
    if similarity >= 0.75:
        result = chain.run(message=message, best_practice=best_practice)
        
    else:
        result = "Mmm,we're not sure about this.Please reach out via [email/social media..etc]"

    
    return result

# 5. Deplot to Streamlit
def main():
    st.set_page_config(
        page_title="HR Response Generator", page_icon=":computer:")

    st.header("HR Advice Generator :computer: ")
    message = st.text_area("Write your query like an employee email")

    if message:
        st.write("Generating reply...")
        result = generate_response(message)
        print(result)
        st.info(result)

if __name__ == '__main__':
    main()
