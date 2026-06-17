import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# PubMed API base URLs
SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def search_pubmed(query, max_results=20):
    # Search PubMed and return list of article IDs
    params = {
        "db": 
        "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    response = requests.get(SEARCH_URL, params=params)
    ids = response.json()["esearchresult"]["idlist"]
    print(f"Found {len(ids)} papers for: '{query}'")
    return ids

def fetch_papers(paper_ids):
    # Fetch full abstract and metadata for each paper ID
    params = {
        "db": "pubmed",
        "id": ",".join(paper_ids),
        "rettype": "abstract",
        "retmode": "xml"
    }
    response = requests.get(FETCH_URL, params=params)
    return response.text

def parse_papers(xml_text):
    # Parse XML response and extract title + abstract
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        # Extract title
        title = article.findtext(".//ArticleTitle", default="No Title")

        # Extract abstract text
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join([a.text for a in abstract_parts if a.text])

        # Extract year
        year = article.findtext(".//PubDate/Year", default="Unknown")

        # Extract authors
        author_list = article.findall(".//Author")
        authors = []
        for author in author_list[:3]:  # First 3 authors only
            last = author.findtext("LastName", default="")
            fore = author.findtext("ForeName", default="")
            authors.append(f"{last} {fore}".strip())

        # Skip papers with no abstract
        if not abstract:
            continue

        papers.append({
            "title": title,
            "abstract": abstract,
            "year": year,
            "authors": ", ".join(authors),
        })

    return papers

def save_papers(papers, path="data/papers.json"):
    # Save fetched papers to JSON file
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {path}")

if __name__ == "__main__":
    # Define medical topics to fetch
    queries = [
        "Type 2 diabetes treatment",
        "metformin side effects elderly",
        "COVID-19 cardiovascular effects",
        "sleep mental health relationship"
    ]

    all_papers = []

    for query in queries:
        ids = search_pubmed(query, max_results=10)
        xml_data = fetch_papers(ids)
        papers = parse_papers(xml_data)
        all_papers.extend(papers)
        print(f"Parsed {len(papers)} papers for '{query}'")

    # Remove duplicates by title
    seen = set()
    unique_papers = []
    for p in all_papers:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique_papers.append(p)

    save_papers(unique_papers)
    print(f"\nTotal unique papers saved: {len(unique_papers)}")