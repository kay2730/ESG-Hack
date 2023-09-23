import streamlit as st
from score import esg, improvement_suggestion, chat
import re
import pandas as pd

# Define functions to get text data for each option
def get_jp_morgan_data(uploaded_pdf):
    if uploaded_pdf:
        scores = esg('JP Morgan')
        suggestions = improvement_suggestion(scores)
        suggestions = re.sub(r'(\d+\.)', r'\n\1', suggestions)
        data = {
            'scores': scores,
            'suggestions': suggestions
        }
        return data

def get_barclays_data(uploaded_pdf):
    if uploaded_pdf:
        scores = esg('Barclays')
        suggestions = improvement_suggestion(scores)
        suggestions = re.sub(r'(\d+\.)', r'\n\1', suggestions)

        data = {
            'scores': scores,
            'suggestions': suggestions
        }
        return data

def get_goldman_sachs_data(uploaded_pdf):
    if uploaded_pdf:
        scores = esg('Goldman Sachs')
        suggestions = improvement_suggestion(scores)
        suggestions = re.sub(r'(\d+\.)', r'\n\1', suggestions)

        data = {
            'scores': scores,
            'suggestions': suggestions
        }
        return data
    
# Chatbot function
def jpmc_chatbot(query):
    scores = esg('JP Morgan')
    answer = chat(scores, query)
    return f"You asked: '{query}'\nChatbot response: {answer}"

def barclays_chatbot(query):
    scores = esg('Barclays')
    answer = chat(scores, query)
    return f"You asked: '{query}'\nChatbot response: {answer}"

def gs_chatbot(query):
    scores = esg('Goldman Sachs')
    answer = chat(scores, query)
    return f"You asked: '{query}'\nChatbot response: {answer}"

# Main Streamlit app
def main():
    # Set the title and the menu options
    st.title("Financial Institutions Information")
    option = st.selectbox("Select an option:", ("JP Morgan", "Barclays", "Goldman Sachs"))

    # Upload a PDF file
    uploaded_pdf = st.file_uploader("Upload a PDF file")

    # Display text data based on the selected option and uploaded PDF
    if option == "JP Morgan":
        data = get_jp_morgan_data(uploaded_pdf)
        
        # Display the 'scores' field as a table
        st.subheader("Scores")
        df = pd.DataFrame(data['scores'].items(), columns=["ESG Parameter", "Score"])
        st.table(df)

        # Display the 'suggestions' field in a separate section
        st.subheader("Suggestions")
        st.write(data['suggestions'])

             # Add a chatbot section
        st.header("Chatbot")
        query = st.text_input("Ask a question:")
        if st.button("Submit"):
            if query:
                answer = jpmc_chatbot(query)
                st.write("Chatbot Response:")
                st.write(answer)

    elif option == "Barclays":
        data = get_barclays_data(uploaded_pdf)
        
        # Display the 'scores' field as a table
        st.subheader("Scores")
        df = pd.DataFrame(data['scores'].items(), columns=["ESG Parameter", "Score"])
        st.table(df)

        # Display the 'suggestions' field in a separate section
        st.subheader("Suggestions")
        st.write(data['suggestions'])

             # Add a chatbot section
        st.header("Chatbot")
        query = st.text_input("Ask a question:")
        if st.button("Submit"):
            if query:
                answer = barclays_chatbot(query)
                st.write("Chatbot Response:")
                st.write(answer)
    elif option == "Goldman Sachs":
        data = get_goldman_sachs_data(uploaded_pdf)

        st.subheader("Scores")
        df = pd.DataFrame(data['scores'].items(), columns=["ESG Parameter", "Score"])
        st.table(df)

        # Display the 'suggestions' field in a separate section
        st.subheader("Suggestions")
        st.write(data['suggestions'])

             # Add a chatbot section
        st.header("Chatbot")
        query = st.text_input("Ask a question:")
        if st.button("Submit"):
            if query:
                answer = gs_chatbot(query)
                st.write("Chatbot Response:")
                st.write(answer)
    else:
        data = ""




if __name__ == "__main__":
    main()
