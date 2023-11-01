import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pickle
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate



st.set_page_config(page_title="Chat with Documents", page_icon="💬")
st.title("💬 Chat with Your Documents")

("Type any question about laws I can help you to overcome your law Problems !")

general_system_template = r"""Your Role is Attorney of law and you have to answer only to the questions related to law. if asked beyond law say you don't know the answer,Given a specific context, please give a short answer to the question, covering the required advices in general and then provide the names all of relevant (even if it relates a bit) products. ----  {context} ---- """
general_user_template = "Question:``` {question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)



# Join all prompts into a single string with a space in between each prompt
# all_prompts = ' '.join(prompts)

@st.cache_resource(ttl="1h")
# Function to read PDF content
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    print(os.path.basename(file_path))  # Change this line
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def configure_retriever(files):
    # Read documents
    # text = read_pdf(file)
     docs = []
     temp_dir = tempfile.TemporaryDirectory()
     for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.read())
        # loader = PyPDFLoader(temp_filepath)
        # docs.extend(loader.load())

        text = read_pdf(temp_filepath)
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        documents = text_splitter.split_text(text=text)
        for i, document_chunk in enumerate(documents):
            # Define metadata for the document chunk
            metadata = {
                "source": file.name,  # You can include any metadata you need
                "chunk_number": i + 1,  # Include the chunk number or any other relevant information
                # Add more metadata fields as needed
            }

            # Create a document instance for the chunk with text and metadata
            document = Document(page_content=document_chunk, metadata=metadata)
            docs.append(document)
    #  combined_text = "\n\n".join(docs)


    

    # Create embeddings and store in vectordb
     embeddings = OpenAIEmbeddings()
     vectordb = FAISS.from_documents(docs, embeddings)

     pickle_folder = "Pickle"
     if not os.path.exists(pickle_folder):
        os.mkdir(pickle_folder)

     pickle_file_path = os.path.join(pickle_folder, f"{file.name}.pkl")

     if not os.path.exists(pickle_file_path):
        with open(pickle_file_path, "wb") as f:
            pickle.dump(vectordb, f)

    # Define retriever
     retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

     return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

load_dotenv()

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True ,combine_docs_chain_kwargs={"prompt": qa_prompt})


# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt})

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[stream_handler])