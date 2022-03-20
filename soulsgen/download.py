import zipfile

import click
from google.cloud import storage


def _get_bucket(bucket: str):
    client = storage.Client()
    return client.get_bucket(bucket)


def unzip(zip_file: str, dst_dir: str):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dst_dir)


def download_torchserve_assets(bucket: str):
    bucket = _get_bucket(bucket)
    file_mappings = (
        ("torchserve-assets/model-store/soulsgen.mar", "serve-handler/soulsgen.mar"),
        (
            "torchserve-assets/assets/soulsgen/corpus/words.txt",
            "serve-handler/assets/words.txt",
        ),
        (
            "torchserve-assets/assets/soulsgen/corpus/vocab.txt",
            "serve-handler/assets/vocab.txt",
        ),
    )
    for src, dst in file_mappings:
        bucket.blob(src).download_to_filename(dst)


def download_torch_model(bucket: str, version: str):
    bucket = _get_bucket(bucket)
    bucket.blob(f"models/soulsgen/{version}/model.pth").download_to_filename(
        "serve-handler/model.pth"
    )


def download_corpus(bucket: str, version: str):
    bucket = _get_bucket(bucket)
    bucket.blob(f"models/soulsgen/{version}/corpus.zip").download_to_filename(
        "corpus.zip"
    )
    unzip("corpus.zip", ".")


@click.command()
@click.argument("resource", type=click.Choice(["model", "handler", "corpus"]))
@click.option("--bucket", "-b", envvar="BUCKET")
@click.option("--version", "-v", default="latest", help="version tag to use")
def download(resource, bucket, version):
    """Download soulsgen assets from a bucket"""
    if resource == "model":
        download_torch_model(bucket, version)
    elif resource == "handler":
        download_torchserve_assets(bucket)
    elif resource == "corpus":
        download_corpus(bucket, version)
