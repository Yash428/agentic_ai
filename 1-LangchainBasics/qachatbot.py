
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os

## Page config

st.set_page_config(
    page_title="Langchain Q&A Chatbot with Groq",
    page_icon="🤖",
    layout="wide"
)
st.title("Langchain Q&A Chatbot with Groq")
st.markdown("""This is a simple Q&A chatbot built using Langchain and Groq. It allows you to ask questions and get answers from the Groq API. The chatbot uses a simple conversation history to maintain context during the conversation.""")


with st.sidebar:
    st.header("Chatbot Settings")
    ## API Key input
    api_key = st.text_input("Enter your Groq API Key", type="password", help="You can get your API key from the Groq dashboard.")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.success("API Key set successfully!")
    
    ## Model selection
    model_name = st.selectbox("Select Groq Model", ["llama-3.1-8b-instant", "gemma2-9b-it"])
    st.info(f"You have selected the {model_name} model.")

    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


## Initialize llm
@st.cache_resource
def get_chain(api_key, model_name):
    """Return a chain combining a prompt template with Groq chat model and output parser."""
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to use the chatbot.")
        return None

    ## Initialize the ChatGroq model
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.7, streaming=True)

    # build prompt template that accepts a single 'input' variable
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided conversation history."),
        ("human", "{input}")
    ])

    # chain prompt -> llm -> parser
    chain = prompt | llm | StrOutputParser()
    return chain

chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please set up the chatbot by providing your API key and selecting a model.")

else:
    ## display the chat messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    ## chat input
    if question:=st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "human", "content": question})
        with st.chat_message("human"):
            st.write(question)

        ## get the answer from the chain
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # the prompt template expects a single key named "input"
                for chunk in chain.stream({"input": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                message_placeholder.markdown(full_response)
