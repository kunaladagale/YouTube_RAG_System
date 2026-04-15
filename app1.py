import streamlit as st
from rag_pipeline import get_transcript, create_chain
st.set_page_config(page_title="YouTube RAG", layout="wide")

st.title("🎥 YouTube RAG Q&A")

# --- Session State ---
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Input ---
video_id = st.text_input("Enter YouTube Video ID")

if st.button("Load Video"):
    with st.spinner("Fetching transcript and building RAG..."):
        try:
            text = get_transcript(video_id)
            chain = create_chain(text)
            st.session_state.chain = chain
            st.success("✅ Ready! Ask questions now.")

        except Exception as e:
            st.error(f"Error: {e}")


# --- Chat UI ---
if st.session_state.chain:
    user_input = st.chat_input("Ask something about the video...")

    if user_input:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_input)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    # Display chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)