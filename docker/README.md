# Getting environment up and running

## Set up local development environment

To get the a local development environment up and running quickly using docker run:
```bash
./bin/install-docker.sh
./bin/build-images.sh
./bin/set-me-up.sh
```

You can skip running `./bin/build-images.sh` if the images are already built.

The `./bin/set-me-up.sh` script will do the following:
- Run a docker container
- Copy the built workspace to `$HOME/docker-dir`
- Stop the container

## Use Docker as a development environment

If you have your workspace locally and want to use the docker machine to run your code do:
```bash
./bin/launch.py --data-dir=$HOME/docker-dir
```