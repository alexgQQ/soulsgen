import os
import stat
import zipfile

import click
from google.cloud import storage


def _get_bucket(bucket: str):
    client = storage.Client()
    return client.get_bucket(bucket)


def zip_dir(zip_file: str, directory: str):
    with zipfile.ZipFile(zip_file, "w") as zip_file:
        for dirname, _, files in os.walk(directory):  # subdirs not needed here
            zip_file.write(dirname)
            for filename in files:
                zip_file.write(os.path.join(dirname, filename))


def upload_file(src: str, dst: str, bucket_name: str):
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(dst)
    blob.upload_from_filename(src)


def upload_corpus(bucket: str, version: str):
    source_file = "corpus.zip"
    zip_dir(source_file, "dataset/build")
    dst = f"models/soulsgen/{version}/{source_file}"
    upload_file(source_file, dst, bucket)


def upload_torchserve_assets(bucket: str):
    os.chmod("serve-handler/soulsgen.mar", 777)
    dst_base = "torchserve-assets/test/model-store"
    upload_file("serve-handler/soulsgen.mar", f"{dst_base}/soulsgen.mar", bucket)
    dst_base = "torchserve-assets/test/assets/soulsgen/corpus"
    for dirname, _, files in os.walk("serve-handler/assets"):
        for _file in files:
            filepath = os.path.join(dirname, _file)
            os.chmod(filepath, 777)
            upload_file(filepath, f"{dst_base}/{_file}", bucket)


@click.command()
@click.argument("resource", type=click.Choice(["handler", "corpus"]))
@click.option("--bucket", "-b", envvar="BUCKET")
@click.option("--version", "-v", default="latest", help="version tag to use")
def upload(resource, bucket, version):
    """Upload soulsgen assets to a bucket"""
    if resource == "handler":
        upload_torchserve_assets(bucket)
    elif resource == "corpus":
        upload_corpus(bucket, version)
