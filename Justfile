alias r := run

run:
	uv run -m uvicorn app:app --host 0.0.0.0 --port 7000

update-fork:
	git pull original dev
	git push
