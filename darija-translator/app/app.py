from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .translator import save_translation_pair, translate_with_source
except ImportError:
    from translator import save_translation_pair, translate_with_source


app = FastAPI(
    title="English to Moroccan Darija Translator API",
    description="Backend API for the React Darija translator interface.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslationRequest(BaseModel):
    text: str = Field(default="", description="English text to translate.")


class TranslationResponse(BaseModel):
    translation: str
    source: str
    is_exact_dataset_match: bool


class CorrectionRequest(BaseModel):
    english: str = Field(default="", description="Original English text.")
    darija: str = Field(default="", description="Correct Moroccan Darija translation.")


class CorrectionResponse(BaseModel):
    saved: bool
    message: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "Darija Translator API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest) -> TranslationResponse:
    result = translate_with_source(request.text)
    return TranslationResponse(
        translation=str(result["translation"]),
        source=str(result["source"]),
        is_exact_dataset_match=bool(result["is_exact_dataset_match"]),
    )


@app.post("/corrections", response_model=CorrectionResponse)
def save_correction(request: CorrectionRequest) -> CorrectionResponse:
    if not request.english.strip() or not request.darija.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter both English and Darija text before saving.",
        )

    saved = save_translation_pair(request.english, request.darija)
    if not saved:
        raise HTTPException(
            status_code=400,
            detail="Please enter both English and Darija text before saving.",
        )

    return CorrectionResponse(
        saved=True,
        message="Saved to dataset.csv. This sentence will use your saved translation next time.",
    )
