Install conda and install requirements using

```console
conda env create -f environment.yml
```

## Example Workflow

1. Download pokemon showdown data (run5-100pages-allgens is already included, so no need to download new data)

```console
python showdown_scraper_all_gen_parallel.py --pages-per-gen 10 10 10 10 10 10 10 10 10 --workers 100 --out-file results_parallel.json
```

- `--pages-per-gen G1 G2 G3 G4 G5 G6 G7 G8 G9`: number of pages to fetch for each generation  
- `--workers N`: number of parallel worker processes  
- `--out-file FILE`: output JSON file for the parallel results  

2. Run train.py and edit the hyperparameters and data path.

```python
#in train.py
data_dir = Path("data/[folder name]")
input_json_path = data_dir / "data.json"
model_path = data_dir / "model.pth"
```

3. Put the model path in predict.py and run with the team.

Example: 
```console
python predict.py --team Pidgey Rattata Ekans Geodude Machop Blissey --model_path data/run5-100pages-allgens/model.pth
```

## Docker
You can run inference on a pretrained model using this:

```console
docker run ghcr.io/maxxu123456/pokemon-teambuilder-ml:1.0.0 --team Pidgey Rattata Geodude Machop Blissey Ekans
```
