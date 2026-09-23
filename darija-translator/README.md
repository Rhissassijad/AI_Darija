# Darija Translator (English -> Moroccan Darija)

A semester-friendly machine translation project with a clean architecture:

- ML pipeline in `src/`
- model artifacts in `models/`
- FastAPI backend in `app/`
- React frontend in `frontend/`
- tests in `tests/`

## Project Structure

```text
darija-translator/
|-- data/
|-- models/
|-- notebooks/
|-- src/
|-- app/
|-- frontend/
|-- tests/
|-- requirements.txt
`-- README.md
```

## Quick Start

1. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Install React frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

3. Run the backend API:

```bash
uvicorn app.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

4. In a second terminal, run the React frontend:

```bash
cd frontend
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

5. Train the model:

```bash
python3 src/model/train.py --epochs 3 --batch_size 8
```

6. Build the React frontend for submission/demo:

```bash
cd frontend
npm run build
```

The production build will be created in `frontend/dist/`.

## API Endpoints

- `GET /health` checks that the API is running.
- `POST /translate` translates English text into Moroccan Darija.
- `POST /corrections` saves a corrected translation to `data/raw/dataset.csv`.

## Grammar-Aware Translation

The translator combines three layers:

1. Exact matches from `data/raw/dataset.csv`.
2. A Darija grammar rule engine for common sentence patterns.
3. The trained model when `models/final_model/` is available.

The rule engine handles common Darija structures including:

- present/habitual verbs with `ka-` plus person prefixes
- future verbs with `ghadi` plus the non-past verb form
- past verbs with person suffixes
- verb negation with `ma-...sh`
- question words such as `chno`, `fin`, `3lash`, `kifach`, and yes/no `wach`
- commands and negative commands
- possession with `dyal`
- `3endi` possession, `khassni` need/must, and `qder` can/could patterns
- basic adjective agreement for feminine subjects

## Dataset Format

`data/raw/dataset.csv` must contain:

```csv
en,darija
Hello,Salam
How are you?,Labas 3lik?
```

## Evaluate

```bash
python3 src/model/evaluate.py
```

## Notes

- Start with a small dataset, then expand gradually.
- For better quality, increase dataset size and train for more epochs.
- Keep checkpoints in `models/checkpoints/`, and export final model to `models/final_model/`.
