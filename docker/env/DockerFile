# Image de base avec PyTorch + CUDA deja compiles -> evite de compiler
# torch avec support GPU a la main, ce qui serait long et fragile.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
# torch est deja dans l'image de base (avec CUDA) -> on evite de le
# reinstaller depuis requirements.txt par-dessus (risque de perdre le
# support CUDA en retelechargeant une version CPU-only par erreur).
RUN grep -v "^torch" requirements.txt > requirements.nocuda.txt && \
    pip install --no-cache-dir -r requirements.nocuda.txt

COPY api.py hybrid_search.py rerank.py ./

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]