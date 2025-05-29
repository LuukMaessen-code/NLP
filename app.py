import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnableParallel
import os

# === Load or initialize FAISS Vector Store ===
if not os.path.exists("faiss_index"):
    os.makedirs("faiss_index")

# Initialize embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Load existing FAISS index if available
faiss_index_file = "faiss_index/index.faiss"
if os.path.exists(faiss_index_file):
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_texts(["initial memory placeholder"], embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# === LLM with conversation memory ===
llm = OllamaLLM(model="llama3")
conversational_llm = ConversationChain(llm=llm, memory=memory)

# === PROMPT 1: Sentiment Reflection Generator ===
reflection_prompt = PromptTemplate(
    input_variables=["entry"],
    template="You wrote the following journal entry:\n\n\"{entry}\"\n\nWhat might this say about your emotional state? Please respond with empathy and insight. Try to keep your responses within 100 words",
)

# === PROMPT 2: Follow-up Prompt Generator ===
followup_prompt = PromptTemplate(
    input_variables=["entry"],
    template="You wrote this journal entry:\n\n\"{entry}\"\n\nSuggest a thoughtful and emotionally intelligent follow-up question for self-reflection. Try to keep your responses within 100 words",
)

# === CHAINS ===
reflection_chain = reflection_prompt | conversational_llm
followup_chain = followup_prompt | conversational_llm

combined_chain = RunnableParallel(
    sentiment=reflection_chain,
    followup=followup_chain
)

# === Streamlit UI ===
st.title("🧠 Mental Health Journal Companion")

entry = st.text_area("Write your journal entry here:")

if st.button("Reflect"):
    if entry.strip():
        with st.spinner("Analyzing and saving your entry..."):
            results = combined_chain.invoke({"entry": entry})

            # Extract raw model text only, excluding prompt or structure
            def extract_response(text):
                if isinstance(text, dict) and "response" in text:
                    return text["response"].strip()
                return str(text).strip().split("response:")[-1].strip()

            sentiment_text = extract_response(results['sentiment'])
            followup_text = extract_response(results['followup'])


            clean_summary = (
                f"### Reflection\n{sentiment_text}\n\n"
                f"### Follow-up\n{followup_text}"
            )

            # Save to vectorstore
            vectorstore.add_texts([clean_summary])


            # Save only the clean text directly into the vectorstore
            vectorstore.add_texts([clean_summary])

            vectorstore.save_local("faiss_index")

            st.markdown("### 🌱 Reflection")
            st.write(results["sentiment"])

            st.markdown("### 💬 Follow-up Question")
            st.write(results["followup"])
    else:
        st.warning("Please enter a journal entry.")

# === Show Memory Log ===
if st.checkbox("🧾 Show Past Reflections"):
    st.subheader("📚 Memory Log")
    for doc in retriever.get_relevant_documents(""):
        st.markdown(doc.page_content, unsafe_allow_html=True)
        st.markdown("---")
