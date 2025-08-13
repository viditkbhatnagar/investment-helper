Setup (Python 3.11 + TensorFlow on macOS ARM)

1) Create TF virtualenv

```
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv_tf
source .venv_tf/bin/activate
pip install -U pip wheel setuptools
pip install tensorflow-macos==2.16.1 tensorflow-metal==1.1.0 keras==3.11.2 keras-tuner==1.4.7
pip install -r requirements.txt
```

2) Run tests and server

```
pytest -q
uvicorn app.main:app --port 8010
```

3) Configure Anthropic in `backend/.env`:

```
ANTHROPIC_API_KEY=your_key
```


