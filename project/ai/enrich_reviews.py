import os 
import json 
import snowflake.connector 
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://foragent07-resource.services.ai.azure.com/openai/v1"
deployment_name = "gpt-5.6-sol"
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)
