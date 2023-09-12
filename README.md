# MLOps Interview Test
### Using Python to run
```
usage: main.py [-h] -d DATA_DIR [-max_seq_len SEQ_MAX_LEN] [-bs BATCH_SIZE]
               [-nw NUM_WORKERS] [--gpus GPUS]
               {train,test} ...

positional arguments:
  {train,test}          train or test mode

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        path to data
  -max_seq_len SEQ_MAX_LEN, --seq_max_len SEQ_MAX_LEN
                        max sequence length permitted
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        dataloader workers
  --gpus GPUS, -gpu GPUS
                        number of gpus used
```
To start a simple training run using default params:
```
python main.py -d 'random_split' 
```
### Run with Docker
Similar to the above except running from a docker image, we first build with:
```
sudo docker build -t instadeep .
```
The container will function as the entry point as the python script. For example, and upon building, you may run training via the command:
```
docker run -dit -p 0.0.0.0:6006:6006 --name main -it --rm instadeep:latest  -d 'random_split' train
```
To test the application with the container, we may run:
```
docker exec -it instadeep bash
```

### Run with shell
The script to perform a single training run is provided :
```
run.sh
```
