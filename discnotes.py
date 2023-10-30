import streamlit as st

def main():
    st.title("Transcript Upload")

    # Upload the transcript file
    uploaded_file = st.file_uploader("Already have a meeting transcript? Upload your file here", type=["txt", "pdf"])

    # Select the meeting type
    meeting_type = st.selectbox("Select your meeting type. This determines how the output is formatted?",
                                ["Discovery Meeting", "Review Meeting", "Investment Plan Meeting"])

    if st.button("Submit"):
        if uploaded_file is not None:
            # Process the uploaded file and meeting type (add your Python code here)
            
            # Example: Print the uploaded file name and meeting type
            st.write(f"Uploaded file: {uploaded_file.name}")
            st.write(f"Meeting type: {meeting_type}")

            # You can perform your desired Python operations here

if __name__ == '__main__':
    main()
