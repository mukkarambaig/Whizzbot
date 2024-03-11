import os
from datetime import datetime
import time
import boto3
from src.vector_stores import load_vector_embeddings, create_vector_embeddings
from langchain.llms.bedrock import Bedrock
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain

bedrock = boto3.client(service_name="bedrock-runtime")

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

def load_knowledge_base():
    """Loads the vector embeddings into the session state."""
    with st.spinner("Loading Knowledge Base..."):
        st.session_state['docsearch'] = load_vector_embeddings()
        st.success("Knowledge Base Loaded!", icon="✅")


def reload_vector_store():
    """Recreates vector embeddings and reloads the knowledge base."""
    create_vector_embeddings()
    load_knowledge_base()


def get_llm():
    """Initializes and returns a Bedrock LLM instance."""
    return Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock,
                   model_kwargs={'max_gen_len': 512, "temperature": 0.2,
                                 "top_p": 0.9})


def retreive_documents(prompt):
    docs = st.session_state['docsearch'].similarity_search(prompt, k=3)
    print(f"Docs: {len(docs)}")

    print(f"======================Retreived Documents======================")
    print(f"*** {prompt} ***")
    for i, doc in enumerate(docs):
        print(f"Doc {i}: {doc.page_content} \n\n\n ")
    return docs


def create_chain():
    llm = get_llm()
    return load_qa_chain(llm)


def generate_llama2_response(prompt):
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a document-based chatbot designed to assist HR professionals by providing precise answers drawn directly from a predefined set of documents. Your role is to analyze and extract relevant information from these documents to address inquiries related to HR practices, policies, and procedures.
                        When presented with a query:
                        1. Search and Respond: Thoroughly search the designated documents to find information directly related to the question. Your response should be strictly based on the document's content, providing specific, detailed answers.
                        2. Citation and Reference: For each response, cite the particular document and section where the information was found, guiding the user to the source and enhancing the response's credibility.
                        3. Handling Unknowns: If the query falls outside the scope of your documents, transparently acknowledge the limitation. Respond with "The information required is not available in my reference documents," and offer a courteous note of apology.
                        4. No External Data: Refrain from integrating any external information or personal insights into your responses. Your answers must be solely document-derived, maintaining the integrity and focus of the chatbot.
                        5. Confidentiality and Sensitivity: Respect the confidentiality of the documents. Do not disclose any sensitive or classified information in your responses.
                        6. Clarity and Brevity: Aim for responses that are clear, concise, and directly address the queries. Your communication should be easy to understand, avoiding unnecessary complexity or ambiguity.
                        7. Structured Responses: Present your answers in bullet points to enhance readability and make the information more accessible. This format will help HR professionals quickly grasp the essential points and apply them effectively.
                        Your mission is to assist HR professionals efficiently by providing accurate, document-based answers, respecting the chatbot's operational guidelines and the sensitive nature of HR-related inquiries."""),
            ("human", "Provided Context: {context}\nUser want to know about: {user_input}"),
        ]
    )

    docs = retreive_documents(prompt)
    messages = chat_template.format_messages(context=docs,user_input=prompt)
    chain = create_chain()
    response = chain.invoke({"input_documents": docs, "question": messages})
    print(f"*****************Response*****************")
    print(f"Response: {response}\n\n\n")
    return response['output_text']


def main():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="💬 Whizzbridge HR Chatbot 13B")
    st.write("Version 2.0")
    for key in st.session_state.keys():
        print(key)
    with st.sidebar:
        st.title('💬 Whizzbridge HR Chatbot')

        if "docsearch" not in st.session_state.keys():
            load_knowledge_base()

        st.button("Reload Knowledge Base", on_click=reload_vector_store)

        # Clear the chat history
        st.button('Clear Chat History', on_click=clear_chat_history)

        # Save the chat history
        st.button('Save Chat History', on_click=save_chat_history)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(placeholder="Type a message...", ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    for char in item:
                        full_response += char
                        placeholder.markdown(full_response)
                        time.sleep(0.005)  # Adding a small delay to simulate typing
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
