#!/usr/bin/env python3
"""
Simple HTTP request utility for Windows environments.
Replaces CURL with a Python-based solution.

Usage:
    python http_request.py GET http://localhost:3000/api/todos
    python http_request.py POST http://localhost:3000/api/todos -d '{"title": "New Todo"}'
    python http_request.py PUT http://localhost:3000/api/todos/1 -d '{"title": "Updated", "completed": true}'
    python http_request.py DELETE http://localhost:3000/api/todos/1
    python http_request.py GET http://localhost:3000/api/todos -H "Authorization: Bearer token"
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_headers(header_list):
    """Parse header arguments into a dictionary."""
    headers = {}
    if header_list:
        for h in header_list:
            if ":" in h:
                key, value = h.split(":", 1)
                headers[key.strip()] = value.strip()
    return headers


def make_request(method, url, headers=None, data=None):
    """Make an HTTP request and return the response."""
    if headers is None:
        headers = {}

    # Set default Content-Type for requests with body
    if data and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    # Encode data if provided
    body = None
    if data:
        if isinstance(data, str):
            body = data.encode("utf-8")
        else:
            body = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            response_headers = dict(response.headers)
            content = response.read().decode("utf-8")
            return status, response_headers, content
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8") if e.fp else ""
        return e.code, dict(e.headers), content
    except urllib.error.URLError as e:
        return None, None, str(e.reason)


def format_response(status, headers, content, verbose=False):
    """Format the response for display."""
    if status is None:
        print(f"Error: {content}", file=sys.stderr)
        return

    if verbose:
        print(f"HTTP {status}")
        print("-" * 40)
        for key, value in headers.items():
            print(f"{key}: {value}")
        print("-" * 40)

    # Try to pretty-print JSON
    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(content)


def main():
    parser = argparse.ArgumentParser(
        description="Simple HTTP request utility for Windows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python http_request.py GET http://localhost:3000/api/todos
  python http_request.py POST http://localhost:3000/api/todos -d '{"title": "New Todo"}'
  python http_request.py PUT http://localhost:3000/api/todos/1 -d '{"completed": true}'
  python http_request.py DELETE http://localhost:3000/api/todos/1
        """,
    )

    parser.add_argument(
        "method",
        choices=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        help="HTTP method",
    )
    parser.add_argument("url", help="Request URL")
    parser.add_argument("-d", "--data", help="Request body (JSON string)")
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        dest="headers",
        help='Request header (format: "Key: Value")',
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show response headers"
    )

    args = parser.parse_args()

    headers = parse_headers(args.headers)

    status, resp_headers, content = make_request(
        args.method, args.url, headers=headers, data=args.data
    )

    format_response(status, resp_headers, content, verbose=args.verbose)

    # Exit with error code if request failed
    if status is None or status >= 400:
        sys.exit(1)


if __name__ == "__main__":
    main()
