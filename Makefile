init:
	python -m venv .venv
	.venv/Scripts/activate

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	python -m pylint --disable=R, C model.ipynb
