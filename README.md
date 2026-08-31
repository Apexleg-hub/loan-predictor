# Loan Approval Predictor

A simple ML classification project: predicts whether a loan application
should be approved, served as an API, deployable to Render.

## Project structure

```
loan-predictor/
├── train_model.py     # Generates data + trains the model, produces model.pkl
├── model.pkl           # The trained model (already generated for you)
├── app.py               # FastAPI app that serves predictions
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## 1. Run it locally

```bash
python3 -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# (optional) retrain the model from scratch:
python train_model.py

# start the API
uvicorn app:app --reload
```

Visit http://127.0.0.1:8000/docs to test it in the browser using FastAPI's
built-in interactive docs (Swagger UI). Click "Try it out" on `/predict`.

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: loan approval predictor"
```

Then on GitHub.com:
1. Click "New repository", name it `loan-predictor`, keep it public or private, don't add a README (you already have one).
2. Copy the commands GitHub shows you under "...or push an existing repository from the command line", something like:

```bash
git remote add origin https://github.com/Apexleg-hub/loan-predictor.git
git branch -M main
git push -u origin main
```

## 3. Deploy to Render

1. Go to https://render.com and sign up / log in (you can sign in with GitHub).
2. Click **New +** → **Web Service**.
3. Connect your GitHub account if prompted, then select the `loan-predictor` repo.
4. Fill in the settings:
   - **Name**: `loan-predictor` (or anything you like)
   - **Region**: choose the one closest to you or your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
5. Click **Create Web Service**.

Render will pull your code, install dependencies, and start the app. This
takes a few minutes the first time. Watch the logs in the Render dashboard;
when you see `Application startup complete`, it's live.

## 4. Test the live deployment

Render gives you a URL like `https://loan-predictor-xxxx.onrender.com`.

Visit `https://loan-predictor-xxxx.onrender.com/docs` to use the same
interactive Swagger UI, now running in the cloud, or test with curl:

```bash
curl -X POST https://loan-predictor-xxxx.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 300000, "credit_score": 720, "loan_amount": 500000, "employment_years": 5, "existing_debt": 50000}'
```

Note: on Render's free tier, the service "sleeps" after 15 minutes of no
traffic and takes ~30-60 seconds to wake back up on the next request. This
is expected behavior, not a bug.

## What each part does (for learning)

- **train_model.py**: creates a synthetic but realistic loan dataset,
  trains a RandomForestClassifier, and saves both the model and the list
  of feature names into `model.pkl` using `joblib`.
- **app.py**: loads `model.pkl` once when the server starts, defines the
  input shape with a Pydantic model (`LoanApplication`), and exposes a
  `/predict` endpoint that returns both a yes/no decision and a
  probability.
- **requirements.txt**: pins exact versions so Render builds the same
  environment you tested locally, avoiding "works on my machine" issues.
- **Render**: a cloud platform that watches your GitHub repo, rebuilds
  automatically whenever you push new commits, and runs your start
  command to keep the app alive.
