# import streamlit as st
# import yaml
# import process_documents
from pypandoc.pandoc_download import download_pandoc
import pypandoc
# import uuid
# import os
# from st_audiorec import st_audiorec
# import assemblyai as aai
import os
import streamlit as st
# import PyPDF2
# from dotenv import load_dotenv
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pickle
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import FAISS


# def load_meeting_types():
#     with open("meetings.yaml", "r") as f:
#         meeting_doc = yaml.load(f, Loader=yaml.FullLoader)
#     return meeting_doc["meetings"]


# @st.cache_data
# def generate_doc(output):
#     random_id = uuid.uuid4().hex
#     file_name = os.path.join("temp", f"output_{random_id}.docx")
#     print(file_name)
#     pypandoc.convert_text(output, "docx", format="md", outputfile=file_name)

#     with open(file_name, "rb") as f:
#         pdf_bytes = f.read()

#     # delete the file
#     os.remove(file_name)

#     return pdf_bytes


def main():
    st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    )

    st.write("# Your AI Meeting! 👋")

    st.markdown(
        """
        'Your AI Meeting' is an all-in-one AI platform to summarize your meetings in the format you need. You can also 'Chat with your Docs' and immediately get answers to any question you have from any document in your knowledge base.
          📝 from uploading your documents 📂 and recording your audio 🎙️, and chatting with your documents 🗣️.
        **👈 Select a option from the sidebar** to explore Your AI Meeting!
    """
    )
#     st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )
#     tab1, tab2,tab3 = st.tabs(["Transcript Upload","Record Audio","Chat with Documents"])
#     meeting_types = load_meeting_types()
#     aai.settings.api_key = f"bb069cceeb9e4a6bab96597fdf1d26b9"

#     with tab1:
#         # if wav_audio_data is not None:
#         #     st.audio(wav_audio_data, format='audio/wav')

#         st.title("Transcript Upload")

#         # Upload the transcript file
#         uploaded_file = st.file_uploader(
#             "Upload your transcript file", type=["txt", "pdf"]
#         )

#         # Select the meeting type
#         selected_meeting_type = st.selectbox(
#             "What type of meeting transcript is this?",
#             meeting_types,
#             format_func=lambda x: x["label"],
#             key=3,
#         )

#         if st.button("Submit", key=4):
#             if uploaded_file is not None:
#                 with st.spinner("Processing..."):
#                     llm_output = process_documents.process_document(
#                         uploaded_file, selected_meeting_type
#                     )
#                     st.download_button(
#                         label="Download as Word Document",
#                         data=generate_doc(llm_output),
#                         file_name="report.docx",
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#                     )

#                 st.write(llm_output)

#     with tab2:
#         st.title("Record Audio")

#         wav_audio_data = st_audiorec()

#         if wav_audio_data is not None:
#             # To play audio in frontend:
#             # st.audio(wav_audio_data.export().read())
#             random_id = uuid.uuid4().hex
#             # To save audio to a file, use pydub export method:
#             # wav_audio_data.export(f"temp/audio_{random_id}.wav", format="wav")
#             # f = open(f"temp/audio_{random_id}.wav", "w")
#             with open(f"temp/audio_{random_id}.wav", mode="bx") as f:
#                 f.write(wav_audio_data)
#                 # with st.spinner("Processing your audio record..."):
                    

#             selected__audio_meeting_type = st.selectbox(
#                 "What type of meeting transcript is this?",
#                 meeting_types,
#                 format_func=lambda x: x["label"],
#                 key=1,
#             )

#             if st.button("Submit", key=2):
#                 with st.spinner("Processing..."):
#                     transcriber = aai.Transcriber()
#                     config = aai.TranscriptionConfig(speaker_labels=True)
#                     audio_file_path = f"temp/audio_{random_id}.wav"
#                     transcript = transcriber.transcribe(
#                         audio_file_path, config=config
#                     )
#                     for utterance in transcript.utterances:
#                         text_to_save = f"Speaker {utterance.speaker}: {utterance.text}"
#                     text_file_path = f"temp/text_{random_id}.txt"
#                     with open(text_file_path, "w") as f:
#                         f.write(text_to_save)
#                     llm_output = process_documents.process_document(
#                         text_file_path, selected__audio_meeting_type
#                     )
#                     st.download_button(
#                         label="Download as Word Document",
#                         data=generate_doc(llm_output),
#                         file_name="report.docx",
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#                     )

#                 st.write(llm_output)
#                 os.remove(audio_file_path)
#                 os.remove(text_file_path)
    
#     with tab3:
#         # Function to read PDF content
#         def read_pdf(file_path):
#             pdf_reader = PyPDF2.PdfReader(file_path)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             return text


#         # Mapping of PDFs
#         pdf_mapping = {
#             'HealthInsurance Benefits': 'health.pdf',
#             'Tax Regime': 'newvsold.pdf',
#             # 'Reinforcement Learning': 'SuttonBartoIPRLBook2ndEd.pdf',
#             # 'GPT4 All Training': '2023_GPT4All_Technical_Report.pdf',
#             # Add more mappings as needed
#         }


#         # Load environment variables
#         load_dotenv()


#         # Main Streamlit app
#         st.title("Query your PDF")
#         st.markdown('''
#         ## About
#         Choose the desired PDF, then perform a query.
#         ''')


#         custom_names = list(pdf_mapping.keys())

#         selected_custom_name = st.selectbox('Choose your PDF', ['', *custom_names])

#         selected_actual_name = pdf_mapping.get(selected_custom_name)

#         if selected_actual_name:
#             pdf_folder = "pdfs"
#             file_path = os.path.join(pdf_folder, selected_actual_name)

#             try:
#                 text = read_pdf(file_path)
#                 st.info("The content of the PDF is hidden. Type your query in the chat window.")
#             except FileNotFoundError:
#                 st.error(f"File not found: {file_path}")
#                 return
#             except Exception as e:
#                 st.error(f"Error occurred while reading the PDF: {e}")
#                 return

#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=150,
#                 length_function=len
#             )

#             # Process the PDF text and create the documents list
#             documents = text_splitter.split_text(text=text)

#             # Vectorize the documents and create vectorstore
#             embeddings = OpenAIEmbeddings()
#             vectorstore = FAISS.from_texts(documents, embedding=embeddings)

#             st.session_state.processed_data = {
#                 "document_chunks": documents,
#                 "vectorstore": vectorstore,
#             }

#             # Save vectorstore using pickle
#             pickle_folder = "Pickle"
#             if not os.path.exists(pickle_folder):
#                 os.mkdir(pickle_folder)

#             pickle_file_path = os.path.join(pickle_folder, f"{selected_custom_name}.pkl")

#             if not os.path.exists(pickle_file_path):
#                 with open(pickle_file_path, "wb") as f:
#                     pickle.dump(vectorstore, f)

#             # Load the Langchain chatbot
#             llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
#             qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

#             # Initialize Streamlit chat UI
#             if "messages" not in st.session_state:
#                 st.session_state.messages = []

#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#             if prompt := st.chat_input("Ask your questions from PDF "f'{selected_custom_name}'"?"):
#                 st.session_state.messages.append({"role": "user", "content": prompt})
#                 with st.chat_message("user"):
#                     st.markdown(prompt)

#                 result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
#                 print(prompt)

#                 with st.chat_message("assistant"):
#                     message_placeholder = st.empty()
#                     full_response = result["answer"]
#                     message_placeholder.markdown(full_response + "|")
#                 message_placeholder.markdown(full_response)
#                 print(full_response)
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})



if __name__ == "__main__":
    download_pandoc()
    # create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")

    main()
