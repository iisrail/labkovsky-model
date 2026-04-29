"""
Modal deployment for Labkovsky AI bot.

Deploys AWQ model with vLLM + RAG pipeline on T4 GPU.

Usage:
    # First time: upload ChromaDB to volume
    modal volume put labkovsky-chroma-db chroma_db /chroma_db --force

    # Deploy the API
    modal deploy modal_deploy.py

    # Test locally
    modal run modal_deploy.py::test_query
"""

import modal

# =============================================================================
# MODAL CONFIG
# =============================================================================

APP_NAME = "labkovsky-bot"
GPU_TYPE = "L4"  # 24GB VRAM, comfortable for 8B AWQ model

# Hugging Face model (upload your model there first)
# Run: huggingface-cli upload YOUR_USERNAME/vikhr-labkovsky-awq models/vikhr-labkovsky-awq
HF_MODEL_ID = "iisrail/vikhr-labkovsky-awq"

# Volume for ChromaDB (model downloaded from HF Hub)
CHROMA_VOLUME = "labkovsky-chroma-db"

# Paths inside container
MODEL_PATH = "/model"
CHROMA_PATH = "/chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Create Modal app
app = modal.App(APP_NAME)

# Create persistent volume for ChromaDB
chroma_volume = modal.Volume.from_name(CHROMA_VOLUME, create_if_missing=True)


def download_model():
    """Download model from Hugging Face Hub at image build time."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to {MODEL_PATH}")


# Container image with all dependencies + model downloaded
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "sentence-transformers>=3.0.0",
        "chromadb>=0.5.0",
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "pydantic>=2.0.0",
        "huggingface_hub>=0.25.0",
    )
    .run_function(download_model)  # Download model at build time
)


# =============================================================================
# SYSTEM PROMPT (matches query_rag.py)
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = (
    "You are psychologist Mikhail Labkovsky. Below are reference materials.\n\n"
    "Book and article fragments contain core principles — use them as the foundation "
    "for your reasoning, but do not copy verbatim.\n"
    "QA examples show how to structure a response — use them as a guide for tone and format.\n\n"
    "{docs}\n\n"
    "CRITICAL RULES:\n"
    "1. Pay close attention to the specific details the user mentions.\n"
    "2. If the user says they already did something, acknowledge it — don't suggest they do it.\n"
    "3. Synthesize principles into YOUR OWN response — never copy text directly.\n\n"
    "Answer in Labkovsky's style: blunt, confident, with tough love if needed. "
    "First explain the root cause, then give concrete steps. "
    "If professional help is needed — say so directly."
)




# =============================================================================
# INFERENCE CLASS (GPU, keeps model loaded)
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        CHROMA_PATH: chroma_volume,
    },
    scaledown_window=300,  # Keep warm for 5 min
    timeout=600,  # 10 min for cold start
)
class LabkovskyInference:
    """Inference class that keeps model loaded between requests."""

    @modal.enter()
    def load_models(self):
        """Load all models on container startup."""
        import chromadb
        from sentence_transformers import SentenceTransformer
        from vllm import LLM, SamplingParams

        print("Loading embedding model (CPU)...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

        print("Loading ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection("labkovsky")
        print(f"  Loaded {self.collection.count()} documents")

        print("Loading vLLM with AWQ model...")
        self.llm = LLM(
            model=MODEL_PATH,
            quantization="compressed-tensors",  # For llm-compressor AWQ
            dtype="half",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,  # Disable CUDA graphs - more stable
        )
        self.tokenizer = self.llm.get_tokenizer()

        print("All models loaded!")

    def retrieve(self, query: str, top_k: int = 2) -> list:
        """Retrieve relevant documents from ChromaDB."""
        # Embed query (E5 format)
        query_text = f"query: {query}"
        query_embedding = self.embed_model.encode(query_text, normalize_embeddings=True)

        # Query book/article docs
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"source_type": {"$ne": "qa_corpus"}},  # Exclude QA for now
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if results['ids'][0]:
            for doc_id, text, meta, dist in zip(
                results['ids'][0], results['documents'][0],
                results['metadatas'][0], results['distances'][0]
            ):
                documents.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "distance": dist,
                })

        # Also get 1 QA doc for style
        qa_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1,
            where={"source_type": {"$eq": "qa_corpus"}},
            include=["documents", "metadatas", "distances"]
        )

        if qa_results['ids'][0]:
            doc_id = qa_results['ids'][0][0]
            documents.append({
                "id": doc_id,
                "text": qa_results['documents'][0][0],
                "metadata": qa_results['metadatas'][0][0],
                "distance": qa_results['distances'][0][0],
            })

        return documents

    @modal.method()
    def ask(
        self,
        question: str,
        temperature: float = 0.5,
        max_tokens: int = 500,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
    ) -> dict:
        """Answer a question using RAG + vLLM."""
        from vllm import SamplingParams

        # Retrieve context
        docs = self.retrieve(question)

        # Format docs for prompt
        def doc_label(doc):
            st = doc.get("metadata", {}).get("source_type", "")
            if st == "qa_corpus":
                return "[QA Example]"
            return "[Book/Article]"

        docs_text = "\n\n".join([
            f"{doc_label(doc)} Doc {i+1}: {doc['text']}"
            for i, doc in enumerate(docs)
        ])

        # Build messages
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(docs=docs_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        answer = outputs[0].outputs[0].text.strip()

        return {
            "question": question,
            "answer": answer,
            "docs_used": len(docs),
        }


# =============================================================================
# WEB ENDPOINT
# =============================================================================

@app.function(
    image=image,
    timeout=600,  # 10 min for cold start
)
@modal.fastapi_endpoint(method="POST")
def ask_labkovsky(request: dict) -> dict:
    """
    Web endpoint for asking questions.

    POST body: {"question": "...", "temperature": 0.5, "max_tokens": 500}
    Returns: {"question": "...", "answer": "...", "docs_used": N}
    """
    question = request.get("question", "")
    if not question:
        return {"error": "No question provided"}

    temperature = request.get("temperature", 0.5)
    max_tokens = request.get("max_tokens", 500)

    # Call the inference class
    inference = LabkovskyInference()
    result = inference.ask.remote(
        question=question,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return result


# =============================================================================
# TEST FUNCTION
# =============================================================================

@app.function()
def test_query():
    """Test the inference pipeline."""
    inference = LabkovskyInference()

    questions = [
        "Почему я не могу уйти от мужа который меня не уважает?",
        "Мой ребёнок не слушается, что делать?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = inference.ask.remote(q)
        print(f"A: {result['answer']}")
        print(f"(docs used: {result['docs_used']})")


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main():
    """Run test queries."""
    test_query.remote()
