"""
Handle Twitter interactions.
Gathers source text and post to twitter
Gives utility function to generate access keys for authorization.
"""

import os
from typing import Dict, Any, Optional
import sys
from requests_oauthlib import OAuth1Session  # pylint: disable=E0401
import requests
import click
from dotenv import load_dotenv


def post_handler(handler: Any, url: str, payload: Optional[Dict] = None):
    """Handler to catch and log POST requests errors"""
    try:
        response = handler(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as err:
        click.echo(f"Http Error Received from {url}: {err}")
    except requests.Timeout as err:
        click.echo(f"Timeout Connecting to {url}: {err}")
    except requests.ConnectionError as err:
        click.echo(f"Connection Error from {url}: {err}")
    except requests.exceptions.JSONDecodeError as err:
        click.echo(f"Error Parsing JSON Response from {url}: {err}")
    except Exception as err:  # pylint: disable=W0703
        click.echo(f"Unknown Error Posting to {url}: {err}")
    sys.exit(1)


def twitter_post(
    message: str,
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_secret: str,
) -> Dict:
    """Post a given message to twitter"""
    twitter_post_url = "https://api.twitter.com/2/tweets"
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_secret,
    )
    return post_handler(oauth.post, twitter_post_url, payload={"text": message})


def generate_access_key(consumer_key: str, consumer_secret: str) -> None:
    """Create a Oauth1 Access Token Keypair for authorization. Requires user interaction"""
    twitter_request_token_url = \
        "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
    twitter_authorization_url = "https://api.twitter.com/oauth/authorize"
    twitter_access_token_url = "https://api.twitter.com/oauth/access_token"

    oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)
    fetch_response = oauth.fetch_request_token(twitter_request_token_url)

    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")

    authorization_url = oauth.authorization_url(twitter_authorization_url)
    click.echo("Please go here to authorize: %s" % authorization_url)
    verifier = click.prompt("Input Access PIN: ")

    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    oauth_tokens = oauth.fetch_access_token(twitter_access_token_url)

    access_token = oauth_tokens["oauth_token"]
    access_token_secret = oauth_tokens["oauth_token_secret"]

    return access_token, access_token_secret


@click.group()
def cli():
    """Authenticate and post tweets with twitter"""
    pass


@cli.command()
@click.option(
    "--consumer-key",
    "-k",
    type=str,
    help="twitter client consumer key, will read from `TWITTER_CONSUMER_KEY` env var",
    required=True,
    envvar="TWITTER_CONSUMER_KEY",
)
@click.option(
    "--consumer-secret",
    "-s",
    type=str,
    help="twitter client consumer secret, will read from `TWITTER_CONSUMER_SECRET` env var",
    required=True,
    envvar="TWITTER_CONSUMER_SECRET",
)
@click.option(
    "--access-token",
    "-a",
    type=str,
    help="twitter client access token, will read from `TWITTER_ACCESS_TOKEN` env var",
    required=True,
    envvar="TWITTER_ACCESS_TOKEN",
)
@click.option(
    "--access-secret",
    "-S",
    type=str,
    help="twitter client access secret, will read from `TWITTER_TOKEN_SECRET` env var",
    required=True,
    envvar="TWITTER_TOKEN_SECRET",
)
@click.option(
    "--torchserve-endpoint",
    "-t",
    type=str,
    help="torchserve inference endpoint, will read from `TORCHSERVE_ENDPOINT` env var",
    required=True,
    envvar="TORCHSERVE_ENDPOINT",
)
def post(
    consumer_key, consumer_secret, access_token, access_secret, torchserve_endpoint
):
    """Post a generated soulsgen sentence to twitter"""
    click.echo(f"Fetching a sentence from {torchserve_endpoint}...")
    sentence = post_handler(requests.post, torchserve_endpoint)
    sentence = sentence["result"]
    click.echo(f"Posting `{sentence}` to twitter...")
    twitter_post(sentence, consumer_key, consumer_secret, access_token, access_secret)
    click.echo("Done!")


@cli.command()
@click.option(
    "--consumer-key",
    "-k",
    type=str,
    help="twitter client consumer key, will read from `TWITTER_CONSUMER_KEY` env var",
    required=True,
    envvar="TWITTER_CONSUMER_KEY",
)
@click.option(
    "--consumer-secret",
    "-s",
    type=str,
    help="twitter client consumer secret, will read from `TWITTER_CONSUMER_SECRET` env var",
    required=True,
    envvar="TWITTER_CONSUMER_SECRET",
)
def auth(consumer_key, consumer_secret):
    """Fetch a access token and secret from twitter, requires user interaction"""
    click.echo("Authenticating with twitter...")
    access_token, access_token_secret = generate_access_key(
        consumer_key, consumer_secret
    )
    click.echo(f"Access Token: {access_token}")
    click.echo(f"Access Token Secret: {access_token_secret}")
    click.echo("Done! Keep these secret!")


def main():
    # Funny hack so .env files are loaded locally in dev
    # and as a standalone
    load_dotenv(f"{os.getcwd()}/.env")
    cli()
