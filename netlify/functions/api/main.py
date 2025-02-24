# netlify/functions/api/main.py
from fastapi import FastAPI
from mangum import Mangum
from controllers.pdf_controller import router as pdf_router  # Import your existing router

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "world from Netlify Function!"}

app.include_router(pdf_router)

handler = Mangum(app)  # Crucial: Wrap your FastAPI app with Mangum


# Remove the uvicorn.run block. Netlify functions don't need this.
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)