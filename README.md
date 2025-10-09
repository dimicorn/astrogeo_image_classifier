# Jet Garden
## Prerequisites
Installation of `task` (not necessary, but recommended)
```bash
curl -sL https://taskfile.dev/install.sh | sh
```

## Installation
### With task
```bash
task install
```
### Without task
Create a virtual environment using `uv venv` or `python3 -m venv .venv`, or any other prefered way.

To install dependencies:
```bash
uv sync --frozen
```
or
```bash
pip install -r requirements.txt
```

## Run
To run a `config.yaml` is needed:
```yaml
fits_path: /path/to/astrogeo
db:
  host: host_name
  dbname: database_name
  user: user_name
  psswd: user_password
```
