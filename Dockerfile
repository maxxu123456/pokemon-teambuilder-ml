FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY . .

ENTRYPOINT ["conda", "run", "-n", "pokemon-ml", "python", "predict.py"]
CMD ["--help"] 