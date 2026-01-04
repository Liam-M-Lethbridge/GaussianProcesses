.PHONY: shell run

shell:
	docker run -it --rm -v "$(PWD)":/app gaussian-processes

run:
	docker run --rm -v "$(PWD)":/app gaussian-processes python main.py
