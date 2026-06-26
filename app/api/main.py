from fastapi import FastAPI

from app.api.routes_business import router as business_router
from app.api.routes_chat import router as chat_router
from app.api.routes_docs import router as docs_router
from app.api.routes_eval import router as eval_router
from app.api.routes_feedback import router as feedback_router
from app.api.routes_knowledge_sources import router as knowledge_sources_router
from app.api.routes_logs import router as logs_router
from app.api.routes_models import router as models_router
from app.api.routes_retrieval import router as retrieval_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="GOV-RAG Enterprise API",
        description="Service API for policy, law, legal-case RAG chat, retrieval, evaluation, and feedback.",
        version="0.1.0",
    )

    app.include_router(business_router)
    app.include_router(chat_router)
    app.include_router(docs_router)
    app.include_router(retrieval_router)
    app.include_router(eval_router)
    app.include_router(feedback_router)
    app.include_router(logs_router)
    app.include_router(models_router)
    app.include_router(knowledge_sources_router)

    return app


app = create_app()
