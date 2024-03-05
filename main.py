import os
from datetime import datetime
import time
import boto3
from vector_stores import load_vector_embeddings, create_vector_embeddings
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
            ("system", """In the context of your task, you are entrusted with a set of documents that serve as your primary reference material. Your objective is to utilize this material to respond to various queries, adhering strictly to the information contained within these documents. As you embark on this task, it's imperative to meticulously search the documents to locate information pertinent to the queries at hand. Your responses should be anchored in the content of these documents, providing detailed and precise answers that reflect the information found.
                        When you encounter a query, your response should be grounded in the documents, citing or referencing specific sections to bolster the accuracy of your answers. If the documents do not contain the necessary information, it is crucial to acknowledge this limitation transparently, either by stating "I do not know" or by offering a polite apology for the absence of the required information.
                        It's essential to refrain from incorporating external information or personal knowledge into your responses. Your answers should be exclusively based on the content of the provided documents. Be mindful of the confidentiality of the documents; if any information is classified as confidential or sensitive, it must not be disclosed in your responses.
                        Strive for clarity and brevity in your communication, ensuring that your responses are straightforward, comprehensible, and directly relevant to the queries. When citing information, clearly reference the specific document and section to guide the inquirer to the source of your response.
                        Your primary goal is to offer helpful, precise, and document-centric answers to each query, adhering to the established guidelines and respecting the constraints of the task. This approach ensures that your responses are not only informative but also respectful of the boundaries set by the nature of the documents and the requirements of your task."""),
            ("human", "{user_input}"),
        ]
    )

    messages = chat_template.format_messages(user_input=prompt)
    docs = retreive_documents(prompt)
    chain = create_chain()
    response = chain.invoke({"input_documents": docs, "question": messages})
    print(f"*****************Response*****************")
    print(f"Response: {response}\n\n\n")
    return response['output_text']


def main():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="💬 Whizzbridge HR Chatbot")
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
                        time.sleep(0.05)  # Adding a small delay to simulate typing
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
