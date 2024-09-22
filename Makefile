include .env

api-dockerfile:
	docker build -t $(PNAME)-api -f ./api/Dockerfile ./api/
