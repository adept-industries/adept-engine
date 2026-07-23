from fastapi import FastAPI

app = FastAPI(
    title="Adept Engine",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"service": "adept-engine"}


@app.get("/health")
def health():
    return {
        "status": "UP"
    }