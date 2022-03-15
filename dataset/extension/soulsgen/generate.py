import os
import time
import asyncio
import aiohttp
import aiofiles
import argparse
import json
from typing import List, Tuple
from tqdm import trange


async def async_req(url: str, payload: dict = {}):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="""
            Script to gather generated sentences from a torchserve instance.
        """,
    )
    parser.add_argument(
        "--source-size",
        required=True,
        type=int,
        help="Source size of the original dataset.",
    )
    parser.add_argument(
        "--output-file", required=True, help="Filepath for the output file"
    )
    parser.add_argument(
        "--host",
        required=True,
        help="Host and port for the torchserve instance.",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        default=4,
        help="Batch size for set of requests to send.",
    )
    parser.add_argument(
        "--sample-size",
        required=False,
        default=0.1,
        help="Sample size of sentences to generate from source text.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    output_file = args.output_file
    source_total = args.source_size 
    target_size = int(source_total * args.sample_size)
    urls = [args.host for _ in range(batch_size)]

    data = []
    for _ in trange(target_size // batch_size):
        data += make_reqs(urls)

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as fobj:
        for line in data:
            fobj.write(line)
            fobj.write("\n")
