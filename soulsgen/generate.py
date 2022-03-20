import asyncio
import json
import os
from typing import List

import click
import aiohttp


async def async_req(url: str, payload: dict = {}):
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=False)
    ) as session:
        response = await session.post(url)
        assert response.status == 200
        data = await response.read()
        return json.loads(data).get("result")


async def gather_req_routines(urls: List[str]):
    download_futures = [async_req(url, {"context": " "}) for url in urls]
    return await asyncio.gather(*download_futures, return_exceptions=True)


def make_reqs(urls: List[str]):
    results = asyncio.run(gather_req_routines(urls))
    return results


def get_file_linecount(filename: str) -> int:
    with open(filename, "r") as file_obj:
        return len(file_obj.readlines())


@click.command()
@click.option(
    "--source-size",
    "-s",
    required=True,
    type=int,
    help="size of the original set of sentences",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(exists=False, dir_okay=False),
    help="file path for where to save the sentences",
)
@click.option(
    "--host",
    "-h",
    required=True,
    help="network endpoint for the torchserve instance",
)
@click.option(
    "--batch-size",
    "-b",
    default=4,
    type=int,
    help="number of requests to make in a cycle",
)
@click.option(
    "--sample-size",
    "-ss",
    default=0.1,
    type=float,
    help="sample size percentage of the original set",
)
def generate(source_size, output_file, host, batch_size, sample_size):
    """Create a set of sentences from a torchserve instance"""
    target_size = int(source_size * sample_size)
    urls = [host for _ in range(batch_size)]
    data = []

    click.echo(f"Creating {target_size} sentences...")

    with click.progressbar(range(target_size // batch_size)) as batches:
        for _ in batches:
            data += make_reqs(urls)

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as fobj:
        for line in data:
            fobj.write(line)
            fobj.write("\n")

    click.echo(f"Saved generated sentences to {output_file}")
