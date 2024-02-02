# Import Statements
import subprocess
import streamlit as st
import requests
import os
from datetime import datetime
import time
from urllib.parse import quote


########### Constants ###########

BASE_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = "/predict"
LOAD_DATA_ENDPOINT = "/load_data"
SET_DATA_PATH_ENDPOINT = "/set_data_path"

########### Constants ###########

########### Functions ###########


# Function to make a prediction
def make_prediction(query, temp, top_k, top_p, max_length):
    prediction_payload = {
        "query": query,
        "temp": temp,
        "top_k": top_k,
        "top_p": top_p,
        "max_length": max_length
    }
    response = requests.post(BASE_URL + PREDICT_ENDPOINT, json=prediction_payload)
    if response.status_code == 200:
        return response.json().get("prediction", "")
    else:
        st.error(f"Error in prediction: {response.status_code}")
        return None


# Function for loading existing data
def load_existing_data():
    response = requests.get(BASE_URL + LOAD_DATA_ENDPOINT)
    if response.status_code == 200:
        st.success("Loaded existing data")
    else:
        st.error(f"Error loading existing data: {response.status_code}")


# Function for setting data path
def set_data_path(selected_folder_path):
    data_path_payload = {"data_path": selected_folder_path}
    response = requests.put(BASE_URL + SET_DATA_PATH_ENDPOINT, json=data_path_payload)
    if response.status_code == 200:
        st.success("Data path updated")
    else:
        st.error(f"Error setting data path: {response.status_code}")


# Function for selecting a file (to derive folder path)
def select_file_ui():
    uploaded_file = st.text_input('Enter directory path')
    if uploaded_file:
        return uploaded_file
    else:
        st.info("Please enter a directory path", icon="ℹ️")


# Function for checking if directory exists and sending request
def check_directory_and_send_request(selected_folder_path):
    if os.path.exists(selected_folder_path) and os.path.isdir(selected_folder_path):
        if not os.listdir(selected_folder_path):  # List is empty, directory is empty
            st.warning("The selected directory is empty.")
        else:
            set_data_path(selected_folder_path)
    else:
        st.error("The selected path does not exist or is not a directory.")


# Function for handling data option
def data_option_handler(data_option):
    if data_option == 'Load existing knowledge base':
        load_existing_data()
    else:
        selected_folder_path = select_file_ui()
        if selected_folder_path:
            check_directory_and_send_request(selected_folder_path)


def save_chat_history():
    # Ensure the 'chats' directory exists
    if not os.path.exists('chats'):
        os.makedirs('chats')

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'chats/{timestamp}.txt'

    # Write the chat history to the file
    with open(filename, 'w') as file:
        for message in st.session_state.messages:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            role = message["role"]
            content = message["content"]
            file.write(f"[{timestamp}] {role.upper()}: {content}\n")

    # Notify the user
    st.success(f'Chat history saved as {filename}')


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


########### Functions ###########


########### Streamlit UI ###########


# App title
st.set_page_config(page_title="🦙💬 Whizzbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Whizzbot')
    st.write('This chatbot is POC of WhizzBridge')

    st.success("Model: TinyLlama 🦙💬")

    st.subheader('Data')
    data_option = st.radio("Choose an option", ('Load existing knowledge base', 'Choose directory for new knowledge base'))
    data_option_handler(data_option)

    st.subheader("Parameters")
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.3, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=10, max_value=50, value=10, step=1)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=1024, value=32, step=8)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "I am Whizzbot. How can I help?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Clear the chat history
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Save the chat history
st.sidebar.button('Save Chat History', on_click=save_chat_history)

# User-provided prompt
if prompt := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prediction = make_prediction(prompt, temperature, top_k, top_p, max_length)
                full_response = ""
                message_placeholder = st.empty()
                # Simulate stream of response with milliseconds delay
                for chunk in prediction.split():
                    full_response += chunk + " "
                    time.sleep(0.005)  # Consider async updates for a more responsive UI
                    message_placeholder.markdown(full_response + "▌")  # Simulating typing

                message_placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})