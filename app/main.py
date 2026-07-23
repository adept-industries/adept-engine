from fastapi import FastAPI

app = FastAPI(
    title="Adept Engine",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Adept Engine Running"}


@app.get("/health")
def health():
    return {"status": "UP"}