import glob
import os
import shutil
import subprocess

import click
from soulsgen.inspect import words, inspect


def create_expanda_config(datasource_dir: str):
    """Set expanda to pull xml files from our datasource"""
    source_files = glob.glob(f"{datasource_dir}/*.xml")
    shutil.copyfile("dataset/expanda.base.cfg", "dataset/expanda.cfg")
    with open("dataset/expanda.cfg", "a") as config:
        for xml in source_files:
            config.write(f"    --soulsgen.parse {xml}\n")


@click.group()
def create():
    """Create various resources for the soulsgen application"""
    pass


@create.command()
@click.option(
    "--src-dir",
    "-d",
    envvar="DATASOURCE_DIR",
    show_default=True,
    default=os.getenv("DATASOURCE_DIR"),
    help="directory of the source xml vocabulary data, defaults to `DATASOURCE_DIR` env var",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def corpus(src_dir):
    """Create a tokenized dataset and report from source files"""
    create_expanda_config(src_dir)
    # Run `expanda build` in the dataset dir to create the base corpus
    subprocess.run("expanda build", shell=True, cwd="dataset")
    words("dataset/build/corpus.raw.txt", "serve-handler/assets/words.txt")
    inspect("dataset/build/corpus.raw.txt", "dataset/build/report.html")


@create.command()
@click.option("--version", "-v", default="latest", help="version tag to use")
def handler(version):
    """Create a service handler for soulsgen and torchserve"""
    # model_path is relative to handler src so it should be ../model.pth
    command = (
        f"torch-model-archiver -f --model-name soulsgen --version {version}"
        f" --serialized-file ../model.pth --model-file model.py --handler handler.py"
        f" --export-path {os.getcwd()}/serve-handler"
    )
    # TODO: Should upgrade python to support this within glob module
    cwd = "serve-handler/src/"
    extra_files = glob.glob(f"{cwd}*.py")
    command += f" --extra-files "
    py_files = []
    for py_file in extra_files:
        py_file = py_file.replace(cwd, "")
        if py_file in ("model.py", "handler.py", "test.py"):
            continue
        py_files.append(py_file)
    command += ",".join(py_files)
    subprocess.run(command, shell=True, cwd="serve-handler/src")


@create.command()
def trainer():
    """Create a GCE instance for GPT2 model training"""
    # The python sdk for creating compute instances is bulky and
    # frankly its easier just to run the bash commands but its yucky
    # maybe one day I'll fully port this
    with open("run_trainer.sh", "w") as script:
        script.write("#!/bin/bash\n")
    command = "envsubst < trainer_script.sh | sed '/^[[:blank:]]*#/d;s/#.*//' >> run_trainer.sh"
    subprocess.run(command, shell=True)
    command = (
        "gcloud compute instances create soulsgen-trainer"
        " --zone=$ZONE --image-family=pytorch-latest-gpu"
        " --image-project=deeplearning-platform-release"
        " --machine-type=n1-highmem-2 --boot-disk-size=50GB"
        " --accelerator='type=nvidia-tesla-t4,count=1'"
        " --metadata='install-nvidia-driver=True'"
        " --maintenance-policy=TERMINATE --scopes=default,storage-rw"
        " --metadata-from-file=startup-script=run_trainer.sh"
    )
    subprocess.run(command, shell=True)
    os.remove("run_trainer.sh")


@create.command()
@click.option(
    "--container-reg",
    "-cr",
    envvar="CONTAINER_REG",
    default=os.getenv("CONTAINER_REG"),
    help="container registry url, will default to `CONTAINER_REG` env var",
    show_default=True,
)
@click.option(
    "--push", "-p", show_default=True, is_flag=True, help="push images to repository"
)
def twitter(container_reg, push):
    """Create a Docker image for the twitter post application"""
    # TODO: was too lazy to port to the docker sdk in the moment
    command = f"docker build -t {container_reg}/soulsgen-twitter:latest ."
    subprocess.run(command, shell=True, cwd="twitter-poster")
    if push:
        command = f"docker push {container_reg}/soulsgen-twitter:latest"
        subprocess.run(command, shell=True)
