alias r := run

run:
	uv run -m uvicorn app:app --host 127.0.0.1 --port 7000
