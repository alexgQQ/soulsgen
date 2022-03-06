# SoulsGen

This is a little side project that I thought would be fun. Essentially its just a dark souls text generator.
This uses text entries from the games and trains agains a GPT2 language model.
This is then accessed through torchserve to create a mostly nonsensical, dark souls-y sentence.
Using a simple cronjob I then automatically post these to twitter.

## Setup

This requires [poetry](https://python-poetry.org/docs/#installation) to use.
A few env configs must be provided as well, I recommend to make a .env file as below:
```bash
# Can take these as is
export GPT2_VOCAB_PATH=serve-handler/assets/vocab.txt
export GPT2_CORPUS_PATH=serve-handler/assets/corpus.txt
export TRAINER_NAME=soulsgen-trainer
export TRAINER_ZONE=us-central1-a

# Replace this with registry named location for ez use
export TWITTER_IMAGE_NAME=soulsgen-twitter:latest

# Specify where to pull models and data
export BUCKET=gs://gcp-bucket-to-use
export DATASOURCE_DIR=/path/to/source/xmls
export TORCHSERVE_ENDPOINT=hostname/to/torchserve

# Specify base twitter auth secrets
export TWITTER_CONSUMER_KEY=<twitter dev api key>
export TWITTER_CONSUMER_SECRET=<twitter dev api secret>

# Fill these in from twitter 'token' command below
# Should only need to generate this once or if changing account
export TWITTER_ACCESS_TOKEN=<generated from twitter bot>
export TWITTER_TOKEN_SECRET=<generated from twitter bot>
```

## Commands

```bash
# Install local python deps
./manage install

# Build dataset
./manage create corpus
# Build dataset
./manage upload corpus
# Deploy a VM for training with deps installed
./manage create trainer
# startup script in install_trainer.sh will autorun training
# tune params there as needed

# Build torchserve assets
./manage create archive
# or download current torchserve assets
./manage download archive

# Test torchserve handler directly
poetry run python serve-handler/src/test.py
# Test it with an actual torchserve instance
./manage run

# Upload if it checks out
./manage upload archive
# Place archive in k8s nfs location

# Generate access tokens
./manage create token
# Test a post
poetry run python twitter-poster/twitter/main.py post
# Build twitter poster image
./manage create image
# Push poster image
./manage upload image
# Deploy cronjob
./manage create cronjob
```

## Dataset

All code dealing with generating the dataset is in the `dataset` dir. 
Gathering the data to work with consists of a few steps:
* Gathering from source game files
* Parse and clean game files
* Tokenize and build corpus train/test set.

### Gathering Data

This data was pulled from the three Dark Souls games installed via steam. This is largely a manual process as of now.
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
    - Gather more data points, I have only used specific sets (item descriptions, game openning dialoge, class descriptions) from DS1-3.
    - Automating this would be awesome, but tricky given how random the file placement is.

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

TODOS:
    - Come up with a way to validate entries are indeed clean, hard to manually validate.

### Build our Corpus (Dataset)

Corpus is a term for a set of language snippets. This is just another reference for the dataset to work off of.
Here I use a corpus building tool [Expanda](https://github.com/affjljoo3581/Expanda) that was specified by the trainer I use.
**This is no longer in active maintenece and is a bit old, but there are not many tools for this**

This tool really just takes a set of files, parses them, tokenizes each word and builds a vocabulary set with a training and testing set.
It is meant to use custom modules to handle specific parsing cases. The `extension.darksouls` module is used to parse these files, clean and tokenize them.
It then feeds the results to exapanda to create the dataset.

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

### Serving

A while back I put up a torchserve instance for doing inference over the web. It works decent enough and is meant to have plugable models so you can serve various models. The infrastructure for this is located on another repo of mine [here](https://github.com/alexgQQ/gke-tutorial).

What is stored here is the handler setup for the edge inference, under `serve-handler`. Torchserve works by packaging a model state file, architecture and a network handler to then be unpacked and executed when needed.

* [Making a custom service](https://github.com/pytorch/serve/blob/master/docs/custom_service.md)
* [Packaing service to an archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve)

The current code for this is an adaptation from the trainer code, stripped down to only what is needed for generation.
**Due to how serve unpacks the files, the related modules are imported in a flat structure for relative imports** 

TODOS:
    - Its a bit slow, ~4s req time, profile and try to improve.
    - (A bit out of scope) Add grafana/prometheus source for serve

### Twitter Bot

This is just a simple outline for how I automate the twitter posts. This all relies on each previous step.
The code is located under `twitter-poster` and is ran as a cronjob in my k8s cluster.

The twitter authentication is probably the trickiest part. Currently this uses their [Oauth1](https://developer.twitter.com/en/docs/authentication/oauth-1-0a) spec and is really just ripped from one of their [examples](https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Manage-Tweets/create_tweet.py).
This requires a set of consumer secrets and access tokens. The consumer secrets are the api key and secret issued by Twitter upon dev app signup, can be found/generated from their dev dashboard.

With these, we can generate an access token pair. I'm not sure when these exactly expire, or even if they do but this cannot be automated.
