"""
Direct CLI for the extraction engine — no LLM agent/router needed for the
common case of "give me the data from this URL". This bypasses the ReAct
agent entirely, so it's faster and has zero dependency on GROQ_API_KEY
unless the page actually needs LLM-assisted field extraction.

Usage:
    python cli.py https://example.com/products
    python cli.py https://example.com/blog --fields "Title, Author, Date" --max-pages 3
    python cli.py https://example.com/products --format json --output my_data.json
"""
import argparse

from src.tools import pipeline


def main():
    parser = argparse.ArgumentParser(description="Intelligent website data extractor.")
    parser.add_argument("url", help="The website URL to extract data from.")
    parser.add_argument("--fields", default="", help="Comma-separated list of fields to prioritize.")
    parser.add_argument("--max-pages", type=int, default=1, help="Number of pages to follow via pagination.")
    parser.add_argument("--output", default=None, help="Explicit output file path.")
    parser.add_argument("--format", default=None, choices=["csv", "json", "jsonl"], help="Output format.")
    args = parser.parse_args()

    field_list = [f.strip() for f in args.fields.split(",") if f.strip()] or None
    path = pipeline.extract(
        args.url,
        fields=field_list,
        output_path=args.output,
        output_format=args.format,
        max_pages=args.max_pages,
    )
    print(f"Done. Data saved to: {path}")


if __name__ == "__main__":
    main()