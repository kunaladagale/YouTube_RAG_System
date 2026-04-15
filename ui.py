import streamlit as st

st.title("🎥 YouTube RAG Q&A")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube URL")

if st.button("Fetch Transcript"):
    # Call your function
    transcript = get_transcript(youtube_url)
    st.session_state['transcript'] = transcript
    st.success("Transcript loaded!")

# Ask question
question = st.text_input("Ask a question")

if st.button("Get Answer"):
    if "transcript" in st.session_state:
        answer = rag_pipeline(st.session_state['transcript'], question)
        st.write("### Answer:")
        st.write(answer)
    else:
        st.warning("Please fetch transcript first")