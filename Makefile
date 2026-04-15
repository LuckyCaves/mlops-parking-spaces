init:
	python -m venv .venv
	.venv/Scripts/activate

install:
	.venv/Scripts/activate
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	python -m pylint --disable=R, C model.ipynb
