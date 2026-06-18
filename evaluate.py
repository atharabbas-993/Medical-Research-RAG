import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline import (
    load_papers,
    convert_to_documents,
    setup_llm,
    setup_embeddings,
    setup_vectorstore,
    setup_hybrid_retriever,
    setup_multi_query,
    build_rag_chain,
    format_docs
)

load_dotenv()

class GeminiDeepEvalModel(DeepEvalBaseLLM):
    # Uses Gemini free tier as judge - separate quota from Groq

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel = None):
        # Gemini returns plain text, parse as JSON if schema given
        response = self.model.invoke(prompt)
        if schema:
            content = response.content.strip()
            # Remove markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return schema(**data)
        return response.content

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        response = await self.model.ainvoke(prompt)
        if schema:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
            return schema(**data)
        return response.content

    def get_model_name(self) -> str:
        return "Gemini 1.5 Flash"

def build_test_cases(rag_chain, multi_query_retriever, test_samples):
    # Run RAG pipeline and build DeepEval test cases
    test_cases = []

    for sample in test_samples:
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        # Limit to top 2 chunks, truncate to save tokens
        retrieved_docs = multi_query_retriever.invoke(question)[:2]
        contexts = [doc.page_content[:250] for doc in retrieved_docs]

        # Get RAG answer
        answer = rag_chain.invoke(question)

        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=ground_truth
        )
        test_cases.append(test_case)

        print(f"✅ Done: {question[:55]}...")

    return test_cases

if __name__ == "__main__":
    print("Setting up pipeline for evaluation...\n")

    # Load data and setup pipeline
    papers = load_papers()
    docs = convert_to_documents(papers)
    llm = setup_llm()
    embeddings = setup_embeddings()
    vectorstore = setup_vectorstore(embeddings)

    hybrid_retriever = setup_hybrid_retriever(docs, vectorstore)
    multi_query_retriever = setup_multi_query(hybrid_retriever, llm)
    rag_chain = build_rag_chain(multi_query_retriever, llm)

    print("✅ Pipeline ready\n")

    # Test samples with ground truth answers
    test_samples = [
        {
            "question": "What are the most effective treatments for Type 2 diabetes?",
            "ground_truth": "Effective treatments for Type 2 diabetes include lifestyle changes, metformin, and other glucose-lowering medications based on recent research."
        },
        {
            "question": "What are the side effects of metformin in elderly patients?",
            "ground_truth": "Metformin can cause gastrointestinal side effects and requires monitoring of kidney function in elderly patients."
        },
    ]

    print("🔄 Running RAG pipeline on test questions...\n")
    test_cases = build_test_cases(rag_chain, multi_query_retriever, test_samples)

    # Setup Gemini model for DeepEval judging
    gemini_model = GeminiDeepEvalModel()

    # Define metrics using Gemini judge
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=gemini_model, verbose_mode=False, async_mode=False),
        AnswerRelevancyMetric(threshold=0.5, model=gemini_model, verbose_mode=False, async_mode=False),
    ]

    print("\n⏳ Running DeepEval evaluation...\n")
    evaluate(test_cases=test_cases, metrics=metrics)

    # Print clean summary
    print("\n" + "=" * 55)
    print("📊 DEEPEVAL RESULTS — MEDICAL RAG PIPELINE")
    print("=" * 55)

    metric_names = ["Faithfulness", "Answer Relevancy"]

    for i, metric in enumerate(metrics):
        score = metric.score
        passed = "✅ PASS" if metric.is_successful() else "❌ FAIL"
        print(f"\n  {metric_names[i]:<20}: {score:.3f}  {passed}")

    print("\n  Score guide: 0.8+ excellent | 0.6-0.8 good | below 0.5 needs work")
    print("=" * 55)