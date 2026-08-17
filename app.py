import streamlit as st
import fitz
import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai


st.set_page_config(
    page_title="IntelliAssist AI",
    layout="wide")


st.title("IntelliAssist AI")
st.write("Smart Document AI Assistant")


with st.sidebar:

    st.header(" IntelliAssist AI")

    st.write("### Features")
    st.write("Document Upload")
    st.write("Semantic Search")
    st.write("RAG Question Answering")
    st.write("Document Summarization")
    st.write("Source Citations")
    st.write("Conversation History")
    st.write("Sentiment & Intent Analysis")

    st.divider()

    st.write("### Project")
    st.write(
        "AI-powered document understanding using "
        "RAG, embeddings and FAISS.")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "Gemini API key was not found. "
        "Please set GEMINI_API_KEY before running the app.")

    st.stop()

client = genai.Client(
    api_key=api_key)


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf"],
    key="pdf_uploader")


if uploaded_file is not None:

    st.success(
        "Document uploaded successfully!")

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf")

    pages = []

    for page_number, page in enumerate(pdf):

        page_text = page.get_text()

        if page_text.strip():

            pages.append(
                {
                    "page": page_number + 1,
                    "text": page_text})


    chunks = []
    chunk_pages = []

    for page in pages:

        words = page["text"].split()

        chunk_size = 150

        for i in range(
            0,
            len(words),
            chunk_size):

            chunk = " ".join(
                words[
                    i:i + chunk_size])

            if chunk.strip():

                chunks.append(chunk)

                chunk_pages.append(
                    page["page"])


    if len(chunks) == 0:

        st.error(
            "No readable text was found in this PDF.")

        st.stop()


    st.success(
        f"Document processed into "
        f"{len(chunks)} chunks.")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True)

    embeddings = embeddings.astype(
        "float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension )

    index.add(
        embeddings)


    st.success(
        f"Created {len(embeddings)} "
        "document embeddings.")


    st.subheader(
        " Document Summary")

    if st.button(
        "Generate Summary"):

        summary_context = "\n\n".join(
            chunks)

        summary_prompt = f"""
You are IntelliAssist AI.

Create a clear and professional summary
of the uploaded document.

Include:

1. Main topic
2. Important objectives
3. Key features
4. Technologies or methods mentioned
5. Important conclusions

Use ONLY information from the document.

DOCUMENT:

{summary_context}
"""

        try:

            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                input=summary_prompt)

            st.subheader( " Document Summary")

            st.write(interaction.output_text)

        except Exception as e:

            st.error(f"Gemini error: {e}")


    st.subheader(
        "Chat with your Document")

    question = st.chat_input(
        "Ask something about your document..." )

    if question:

        question_embedding = (
            embedding_model.encode(
                [question],
                convert_to_numpy=True))

        question_embedding = (
            question_embedding.astype(
                "float32"))

        number_of_results = min(
            3,
            len(chunks))

        distances, indices = index.search(
            question_embedding,
            number_of_results)


        retrieved_chunks = []

        for i in indices[0]:

            retrieved_chunks.append(
                chunks[i])


        context = "\n\n".join(
            retrieved_chunks)


        analysis_prompt = f"""
Analyze this user question.

QUESTION:
{question}

Return exactly:

Sentiment: Positive, Neutral, or Negative
Intent: A short description of what the user wants
"""

        try:

            analysis_interaction = (
                client.interactions.create(
                    model="gemini-3.6-flash",
                    input=analysis_prompt))

            analysis = (analysis_interaction.output_text)

        except Exception:

            analysis = ("Sentiment: Neutral\n"
                "Intent: Information request")


        prompt = f"""
You are IntelliAssist AI,
a smart document assistant.

Answer the user's question using ONLY
the retrieved information from the document.

If the answer cannot be found in the
retrieved information, say:

"I could not find the answer in the uploaded document."

RETRIEVED DOCUMENT SECTIONS:

{context}

USER QUESTION:

{question}
"""

        try:

            interaction = (client.interactions.create(model="gemini-3.6-flash",
                    input=prompt))

            answer = interaction.output_text


        except Exception as e:

            answer = (f"Unable to generate an AI answer.\n\n"
                f"Error: {e}")


        st.session_state.chat_history.append( {"question": question,
                "answer": answer})


        with st.chat_message("user"):

            st.write(question)


        with st.chat_message("assistant"):

            st.write(answer)


        st.subheader( "Question Analysis")

        st.write(analysis)


        st.subheader(
            "Retrieved Sources")

        for source_number, i in enumerate(
            indices[0],
            start=1):

            with st.expander(
                f"Source {source_number} — "
                f"Page {chunk_pages[i]}"):

                st.write(chunks[i])

    if st.session_state.chat_history:

        st.subheader(
            "Conversation History")

        for chat in (
            st.session_state.chat_history):

            st.markdown(f"**You:** {chat['question']}")

            st.markdown(f"**IntelliAssist:** "
                f"{chat['answer']}")

            st.divider()


else:

    st.info("Please upload a PDF to begin.")