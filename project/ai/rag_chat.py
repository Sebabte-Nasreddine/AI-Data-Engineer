import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

from hybrid_search import hybrid_search
from rerank import rerank

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
PREFETCH_LIMIT = 50   # candidats recuperes par la recherche hybride (recall)
TOP_K = 8              # candidats gardes apres reranking (precision, contexte du LLM)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------------------------------
# Retrieval : hybride (Qdrant) + reranking (cross-encoder local)
# --------------------------------------------------------------------------

def retrieve(question: str, filters: dict | None):
    candidates = hybrid_search(question, filters=filters, limit=PREFETCH_LIMIT)
    if not candidates:
        return []
    return rerank(question, candidates, top_k=TOP_K)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def ask_llm(question: str, top_reviews: list) -> str:
    context = ""
    for r in top_reviews:
        p = r.payload
        context += f" ({p['city']}, {p['rating']} stars, {p['sentiment_label']}) {p['text']}\n"

    system_prompt = (
        "Answer ONLY using the customer reviews provided. "
        "Be concise. If the reviews don't cover it, say so."
    )
    user_prompt = f"Question: {question}\n\nReviews:\n{context}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# UI Streamlit
# --------------------------------------------------------------------------

st.title("Chat with your Zomato Reviews")
st.caption(f"Recherche hybride (dense + BM25) sur Qdrant, reranking cross-encoder, top-{TOP_K} -> {CHAT_MODEL}")

with st.sidebar:
    st.header("Filtres")
    city = st.text_input("Ville (exact, ex: Casablanca)", value="")
    sentiment = st.selectbox("Sentiment", ["", "positive", "negative", "neutral"])
    rating_range = st.slider("Note", min_value=1, max_value=5, value=(1, 5))

question = st.text_input(
    "Ask a question about your reviews:",
    placeholder="e.g. What are the most common complaints about delivery?",
)

if question:
    filters: dict = {}
    if city:
        filters["city"] = city
    if sentiment:
        filters["sentiment_label"] = sentiment
    if rating_range != (1, 5):
        filters["rating_min"], filters["rating_max"] = rating_range

    top_reviews = retrieve(question, filters or None)

    if not top_reviews:
        st.warning("Aucune review ne correspond à cette question avec ces filtres.")
    else:
        answer = ask_llm(question, top_reviews)

        st.markdown("**Answer:**")
        st.write(answer)

        with st.expander(f"Reviews utilisées ({len(top_reviews)})"):
            for r in top_reviews:
                p = r.payload
                st.markdown(
                    f"- **{p['restaurant_name']}** ({p['city']}, {p['rating']}★, "
                    f"{p['sentiment_label']}) — score rerank: {r.rerank_score:.3f}\n"
                    f"  \n  {p['text']}"
                )