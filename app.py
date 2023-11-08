
# from pypandoc.pandoc_download import download_pandoc
# import pypandoc

import os
import streamlit as st


def main():
    st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    )

    st.write("# Your AI Attorney! 👋")

    st.markdown(
        """
        'Your AI Attorney is an all-in-one AI Agent to provide leagal advices by submitting him documents 📂 and you can also 🗣️ 'Chat with that Docs' and immediately get answers to any question related to the documents in your knowledge base.
        **👈 upload your docs from the sidebar** to explore Your AI Attorney!
    """
    )

if __name__ == "__main__":
    # download_pandoc()
    # create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")

    main()
