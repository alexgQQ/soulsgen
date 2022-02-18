"""
Handle Twitter interactions.
Gathers source text and post to twitter
Gives utility function to generate access keys for authorization.
"""

from typing import Dict, Any, Optional
import logging
import sys
import argparse
from requests_oauthlib import OAuth1Session  # pylint: disable=E0401
from pydantic import BaseSettings  # pylint: disable=E0401
import requests

logging.basicConfig(level="INFO")


class BaseConfig(BaseSettings):  # pylint: disable=R0903
    """Basic configuration required for any action"""

    twitter_consumer_key: str
    twitter_consumer_secret: str


class GenerateConfig(BaseConfig):  # pylint: disable=R0903
    """Configuration required for Twitter authorization"""

    twitter_request_token_url: str = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
    twitter_authorization_url: str = "https://api.twitter.com/oauth/authorize"
    twitter_access_token_url: str = "https://api.twitter.com/oauth/access_token"


class PostConfig(BaseConfig):  # pylint: disable=R0903
    """Configuration required for Twitter posting"""

    torchserve_endpoint: str
    twitter_post_url: str = "https://api.twitter.com/2/tweets"

    twitter_access_token: str
    twitter_token_secret: str


def post_handler(handler: Any, url: str, payload: Optional[Dict] = None):
    """Handler to catch and log POST requests errors"""
    try:
        response = handler(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as err:
        logging.info("Http Error Received from %s: %s", url, err)
    except requests.Timeout as err:
        logging.info("Timeout Connecting to %s: %s", url, err)
    except requests.ConnectionError as err:
        logging.info("Connection Error from %s: %s", url, err)
    except requests.exceptions.JSONDecodeError as err:
        logging.info("Error Parsing JSON Response from %s: %s", url, err)
    except Exception as err:  # pylint: disable=W0703
        logging.info("Unknown Error Posting to %s: %s", url, err)
    sys.exit(1)


def fetch_souls_text() -> Dict:
    """Fetch generated Dark Souls text from torchserve instance"""
    config = PostConfig()
    logging.info("Fetching generated text from %s", config.torchserve_endpoint)
    return post_handler(requests.post, config.torchserve_endpoint)


def twitter_post(message: str) -> Dict:
    """Post a given message to twitter"""
    config = PostConfig()
    oauth = OAuth1Session(
        config.twitter_consumer_key,
        client_secret=config.twitter_consumer_secret,
        resource_owner_key=config.twitter_access_token,
        resource_owner_secret=config.twitter_token_secret,
    )
    logging.info("Posting twitter message to %s", config.twitter_post_url)
    return post_handler(oauth.post, config.twitter_post_url, payload={"text": message})


def generate_access_key() -> None:
    """Create a Oauth1 Access Token Keypair for authorization. Requires user interaction"""
    config = GenerateConfig()

    oauth = OAuth1Session(
        config.twitter_consumer_key, client_secret=config.twitter_consumer_secret
    )
    fetch_response = oauth.fetch_request_token(config.twitter_request_token_url)

    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")

    authorization_url = oauth.authorization_url(config.twitter_authorization_url)
    print("Please go here and authorize: %s" % authorization_url)
    verifier = input("Paste the PIN here: ")

    oauth = OAuth1Session(
        config.twitter_consumer_key,
        client_secret=config.twitter_consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    oauth_tokens = oauth.fetch_access_token(config.twitter_access_token_url)

    access_token = oauth_tokens["oauth_token"]
    access_token_secret = oauth_tokens["oauth_token_secret"]

    print(f"TWITTER_ACCESS_TOKEN={access_token}")
    print(f"TWITTER_TOKEN_SECRET={access_token_secret}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Handle twitter operations for soulsgen."
    )
    parser.add_argument(
        "action",
        choices=["post", "token"],
        type=str,
        help="What operation to run, 'post' to generate a twitter post, \
            'token' to generate a access token for twitter",
    )
    args = parser.parse_args()

    if args.action == "token":
        logging.info("Generating twitter access token")
        generate_access_key()
    elif args.action == "post":
        logging.info("Beginning soulsgen auto twitter post")
        sentence = fetch_souls_text()
        logging.info("Fetched message '%s' from source", sentence["result"])
        twitter_post(sentence["result"])
        logging.info("Soulsgen post complete!")
