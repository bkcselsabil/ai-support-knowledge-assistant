import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

load_dotenv()

base_url = os.getenv("CONFLUENCE_BASE_URL")
email = os.getenv("CONFLUENCE_EMAIL")
api_token = os.getenv("CONFLUENCE_API_TOKEN")

if not all([base_url, email, api_token]):
    raise ValueError("Missing Confluence credentials in .env")

url = f"{base_url}/wiki/rest/api/content/917505"

response = requests.get(
    url,
    auth=HTTPBasicAuth(email, api_token),
    headers={"Accept": "application/json"},
    params={"expand": "space,version,body.storage"},
    timeout=20,
)

print("Status code:", response.status_code)

if response.ok:
    page = response.json()
    print("Connected successfully!")
    print("Page title:", page.get("title"))
    print("Space key:", page.get("space", {}).get("key"))
    html_content = page.get("body", {}).get("storage", {}).get("value", "")

    soup = BeautifulSoup(html_content, "html.parser")
    plain_text = soup.get_text(separator="\n", strip=True)

    print("Page text:")
    print(plain_text)
else:
    print("Request failed. Response:", response.text[:500])