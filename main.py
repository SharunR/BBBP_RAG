import pickle

import fastapi
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from predictor import extract_features

with open('bbbp_xgb.pkl', 'rb') as f:
    model = pickle.load(f)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)
app=FastAPI(title="BBBP penetration",version="1")
class smileRequest(BaseModel):
    smile:str
class QuestionRequest(BaseModel):
    question: str
@app.post('/Upload_smile')
def process_smile(data:smileRequest):
    smlist=list(data.smile)
    result,_=extract_features(smlist)

    pred=model.predict(result)
    prob=model.predict_proba(result)
    return {
    "smile_vector": result.tolist(),
    "prediction": pred.tolist(),
    "probability": prob.tolist()
}
@app.post("/ask")
def ask_question(data: QuestionRequest):

    results = retriever.invoke(
        data.question
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    sources = [
        doc.metadata
        for doc in results
    ]

    return {
        "question": data.question,
        "answer": results[0].page_content,
        "source": results[0].metadata
    }


