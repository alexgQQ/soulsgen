import glob
import os
import click
import docker
from dotenv import load_dotenv

# TODO: Maybe use a differnet import pattern to add these?
from soulsgen.inspect import words_cli, inspect_cli
from soulsgen.generate import generate
from soulsgen.create import create
from soulsgen.download import download
from soulsgen.upload import upload

commands = (
    words_cli,
    inspect_cli,
    generate,
    create,
    download,
    upload,
)


@click.group()
def cli():
    """A set of utilities for training the soulsgen service"""
    pass


@cli.command()
def serve():
    """Run a torchserve container with the current service handler"""
    client = docker.from_env()
    ports = {
        "8080/tcp": "8080",
        "8081/tcp": "8081",
        "8082/tcp": "8082",
        "7070/tcp": "7070",
        "7071/tcp": "7071",
    }
    volumes = [f"{os.getcwd()}/serve-handler:/home/model-server/mnt"]
    environment = [
        "GPT2_VOCAB_PATH=/home/model-server/mnt/assets/vocab.txt",
        "GPT2_CORPUS_PATH=/home/model-server/mnt/assets/words.txt",
    ]
    command = (
        "torchserve --start --ts-config /home/model-server/config.properties"
        " --model-store=/home/model-server/mnt/ --models=soulsgen.mar"
    )
    client.containers.run(
        "pytorch/torchserve:latest-cpu",
        command=command,
        auto_remove=True,
        environment=environment,
        detach=True,
        name="serve",
        ports=ports,
        volumes=volumes,
    )


@cli.command()
def clean():
    paths = (
        "serve-handler/assets/*",
        "dataset/build/*",
        "dataset/expanda.cfg",
        "**/*.pth",
        "**/*.mar",
    )
    for path in paths:
        for _file in glob.glob(path):
            os.remove(_file)


def run():
    load_dotenv()
    for cmd in commands:
        cli.add_command(cmd)
    cli()
