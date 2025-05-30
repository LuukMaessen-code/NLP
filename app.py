import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os

# === Load or initialize FAISS Vector Store ===
if not os.path.exists("faiss_index"):
    os.makedirs("faiss_index")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
faiss_index_file = "faiss_index/index.faiss"

if os.path.exists(faiss_index_file):
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts(["initial memory placeholder"], embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = OllamaLLM(model="llama3")

# === Prompt Templates ===
reflection_prompt = PromptTemplate(
    input_variables=["entry"],
    template="You wrote the following journal entry:\n\n\"{entry}\"\n\nWhat might this say about your emotional state? Please respond with empathy and insight. Try to keep your responses within 100 words",
)

followup_prompt = PromptTemplate(
    input_variables=["entry"],
    template="You wrote this journal entry:\n\n\"{entry}\"\n\nSuggest a thoughtful and emotionally intelligent follow-up question for self-reflection. Try to keep your responses within 100 words",
)

# === Streamlit UI ===
st.title("🧠 Mental Health Journal Companion")
entry = st.text_area("Write your journal entry here:")

# === Helper Function ===
def clean_output(text):
    """Ensure only clean response text is used."""
    return str(text).strip().split("response:")[-1].strip()

if st.button("Reflect"):
    if entry.strip():
        with st.spinner("Analyzing your entry..."):

            # Generate responses
            sentiment_prompt = reflection_prompt.format(entry=entry)
            followup_prompt_text = followup_prompt.format(entry=entry)

            sentiment_response = clean_output(llm.invoke(sentiment_prompt))
            followup_response = clean_output(llm.invoke(followup_prompt_text))

            # Combine clean text for storage
            memory_entry = (
                f"### Reflection\n{sentiment_response}\n\n"
                f"### Follow-up\n{followup_response}"
            )

            # Save only clean summary
            vectorstore.add_texts([memory_entry])
            vectorstore.save_local("faiss_index")

            # Display clean results
            st.markdown("### 🌱 Reflection")
            st.write(sentiment_response)

            st.markdown("### 💬 Follow-up Question")
            st.write(followup_response)
    else:
        st.warning("Please enter a journal entry.")

# === Show Memory Log (Only Clean Entries) ===
if st.checkbox("🧾 Show Past Reflections"):
    st.subheader("📚 Memory Log")
    for doc in retriever.get_relevant_documents(""):
        st.markdown(doc.page_content, unsafe_allow_html=True)
        st.markdown("---")
