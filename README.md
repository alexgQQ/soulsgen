# SoulsGen

This is a little side project that I thought would be fun. Essentially its just a dark souls text generator.
This uses text entries from the games and trains agains a GPT2 language model.
This is then accessed through torchserve to create a mostly nonsensical, dark souls-y sentence.
Using a simple cronjob I then automatically post these to twitter.

## Install

This requires [poetry](https://python-poetry.org/docs/#installation) to use.
There are some poetry bugs around Cython packages, so for now they can be
worked around by manually installing. Additionally the spacy english core
module must be downloaded.

```bash
poetry install  # This will fail but installs most packages
poetry run pip install Cython
poetry install
poetry run python -m spacy download en_core_web_sm
```

## Run

The commands will be installed as a cli script and offer various utils for various operations.
For a general overview:
```bash
poetry run soulsgen --help
poetry run twitter --help
```

### Env Setup

The cli will take these arguments, but many can be loaded as env vars or via a .env file.
```
CONTAINER_REG=container.registry.host
BUCKET=gcp-bucket-to-use
DATASOURCE_DIR=/path/to/source/xmls
TORCHSERVE_ENDPOINT=hostname/to/torchserve
TWITTER_CONSUMER_KEY=<twitter dev api key>
TWITTER_CONSUMER_SECRET=<twitter dev api secret>
TWITTER_ACCESS_TOKEN=<generated from twitter command>
TWITTER_TOKEN_SECRET=<generated from twitter command>
```

## Dataset

All code dealing with generating the dataset is in the `dataset` dir. 
Gathering the data to work with consists of a few steps:
* Gathering from source game files
* Parse and clean game files
* Tokenize and build corpus train/test set.

### Gathering Data

This data was pulled from the three Dark Souls games and Sekiro installed via steam. This is largely a manual process as of now.
An outline of the steps for reference:

* Unpack base game files
    * If using DS1 REMASTERED, this is not needed
    * DS2, DS3 and Sekiro can be unpacked with [UXM](https://github.com/JKAnderson/UXM)
    * DS1 PTDE can be unpacked with [UDSFM](https://www.nexusmods.com/darksouls/mods/1304)

* Find and unpack text files, usually in "msg/<language>" dirs
    * These can be unpacked with [Yaber](https://github.com/JKAnderson/Yabber).
    * If the files is .DCX then the specific Yaber executable must be used

* In the unpacked dirs, find and unpack the needed .FMG files with Yaber.
    * Refer to this [site](http://soulsmodding.wikidot.com/format:fmg#toc1) for some mappings.

The resulting xml file will be the text entries for various entities. These files contain menu messages, npc dialogue, item names etc. Every string literal.

TODOS:
    * Automating this would be awesome, but tricky given how random the file placement is.

### Parse this crazy XML

A good chunk of manual digging had to be done to identity some pieces of interest. Some notes for parsing these:
	* Entries are related by an `id` tag thats specific to each file pairing. Can be used to link entries.
	* Some names contain upgrades like "Estus Flask +2", some relatd entries have duplicate descriptions.
	* Entries for body parts like `Female Head`.
	* Blank entries tend to be "%null%" but can sometimes be "no_text"
	* Weird name entries for `Carvings`?
	* Some weapon descriptions have an `Attack Type: <type>\n` and `Weapon Type: <type>\n` entry for their types
	* Some items are the scrolls for spells(duplicates), names formatted like `Miracle: `, `Pyromancy: `, `Sorcery: `
	* Armor desc have sections with a pattern of "<piece of armor> worn by the bad guys", could make generator averse
    * Some descriptions are specific to actionable items, like "Skill: "

The module `extension` will handle a lot of this parsing and cleaning. Check out its source for implementation details.

### Build our Corpus (Dataset)

Corpus is a term for a set of language snippets. This is just another reference for the dataset to work off of.
Here I use a corpus building tool [Expanda](https://github.com/affjljoo3581/Expanda) that was specified by the trainer I use.
**This is no longer in active maintenece and is a bit old, but there are not many tools for this**

This tool really just takes a set of files, parses them, tokenizes each word and builds a vocabulary set with a training and testing set.
It is meant to use custom modules to handle specific parsing cases. The `soulsgen.parse` module is used to parse these files, clean and tokenize them.
It then feeds the results to exapanda to create the dataset. After parsing, we can run a short report on some of the words with the `soulsgen inspect` command. This relays information on the sentence lengthe and structure for a quick glance at the dataset.

TODOS:
    - Fork and update expanda to work with newer tokenizer libraries.

## Training

For training I opted to use this repo as a starting point https://github.com/affjljoo3581/GPT2.git. It more recent, fast and uses torch.
Training is done on GCP VMs with GPU availability. This repo works great out of the box for it.

My current model took about 6 hours of training time. This ran for 10000 iterations.
If the trained model seems good, upload it to the defined bucket location.

TODO:
    - Automate this, can pull and push results to predefined locations and can deploy startup code to run this automatically.
    - Fork base trainer repo and update areas where cpu deviuce fails(is meant to run with cuda gpu support), add in my generate sentence function too
    - Evaluate model parameters, not sure if any dimension tuning would be helpfull.

## Generation and Serving

### Generation

The model can't really generate a sentence without some context to start with. We must provide some starting point so it can build a sentence.
This is done by gathering a unique set of words from the text entries and picking a random one to use for the sentence start. This is a bit hacky but from testing I found that:
* using nothing as a context creates pure, illegible garble
* using unfamiliar words produces mostly illegible garble
* using words that start a sentence from the dataset, usually relays an exact sentence from the dataset, not very generative

By trial and error, the most legible garble came from picking a random, familiar word to kick the sentence off.
**Note this is not taken from the vocab.txt set as those are tokenized and not all readable friendly**
So generation requires we generate this data seperatly. This is done in the `extension.words` module for the dataset. It will pull the unique words from
the raw sentences and save them to a text file.

Another approach I tried was to use a random letter, generate the sentence, and find the closest word it tried to make from that first letter from my vocab set using a [Levenshtein](https://pypi.org/project/Levenshtein/) distance. Was interesting but felt needlessly complicated.

### Evaluation

While training will give metrics on how well a model is doing, I still want to compare generated text with the source corpus.
Once a model is trained, I simply generate a sample set of sentences with torchserve and run the same lexical report as I would for the source text.
This gives me a good idea if the sentence breakdown is similar, however the process is a bit slow. Running this on CPU alone gets ~1.8 seconds/sentence.
I can get much better performance running this on my GPU through my windows machine.

### Installing for Windows

Run through this guide for [torchserve on windows.](https://github.com/pytorch/serve/blob/master/docs/torchserve_on_win_native.md)
A few notes:
* The `nvidia-smi.exe` exec provided by nvidia is not in the referenced location, just do a file search for it.
* We must explicitly install the latest cuda libraries that come with conda, so when installing deps do:
    ```
    python .\ts_scripts\install_dependencies.py --environment=prod --cuda cu111
    ```

After that, copy over the soulgen service handler and the dataset assets. Then create a config.properties file with `number_of_gpu=1`.
There should be a dir structure like:
```
.
├── model_store
│   └── soulsgen.mar
├── assets
│   ├── vocab.txt
|   └── words.txt
└── config.properties
```

Run the following in the conda shell
```batch
SET GPT2_VOCAB_PATH=assets\vocab.txt
SET GPT2_CORPUS_PATH=assets\words.txt
torchserve --start --ncs --model-store model_store --ts-config config.properties --model soulsgen.mar
```

There may be some error output for the metrics server, an odd formatting bug, but does not interfere with inference.
This will, by default, register 4 worker processes for the model and sending batch jobs to match the worker count
had the best performance, around 0.5 seconds/sentence.


### Serving

A while back I put up a torchserve instance for doing inference over the web. It works decent enough and is meant to have plugable models so you can serve various models. The infrastructure for this is located on another repo of mine [here](https://github.com/alexgQQ/gke-tutorial).

What is stored here is the handler setup for the edge inference, under `serve-handler`. Torchserve works by packaging a model state file, architecture and a network handler to then be unpacked and executed when needed.

* [Making a custom service](https://github.com/pytorch/serve/blob/master/docs/custom_service.md)
* [Packaing service to an archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve)

The current code for this is an adaptation from the trainer code, stripped down to only what is needed for generation.
**Due to how serve unpacks the files, the related modules are imported in a flat structure for relative imports** 

### Twitter Bot

This is just a simple outline for how I automate the twitter posts. This all relies on each previous step.
The code is located under `twitter-poster` and is ran as a cronjob in my k8s cluster.

The twitter authentication is probably the trickiest part. Currently this uses their [Oauth1](https://developer.twitter.com/en/docs/authentication/oauth-1-0a) spec and is really just ripped from one of their [examples](https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Manage-Tweets/create_tweet.py).
This requires a set of consumer secrets and access tokens. The consumer secrets are the api key and secret issued by Twitter upon dev app signup, can be found/generated from their dev dashboard.

With these, we can generate an access token pair. I'm not sure when these exactly expire, or even if they do but this cannot be automated.
