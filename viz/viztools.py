import mimetypes
import os
import uuid
from urllib.parse import urljoin, urlparse

import click
from bs4 import BeautifulSoup
from google.cloud import storage


@click.group()
def cli():
    """CLI tool for managing Google Cloud Storage uploads."""
    pass


@cli.command()
@click.argument("html_file_path", type=click.Path(exists=True))
@click.argument("bucket_name", type=str)
@click.option("--destination-folder", default="", help="Destination folder in the bucket.")
@click.option("--public/--private", default=True, help="Make uploaded files publicly accessible.")
def upload_html_and_assets(html_file_path: str, bucket_name: str, destination_folder: str = "", public: bool = True):
    """
    Uploads a local HTML file and its corresponding assets folder to Google Cloud Storage.
    Before uploading, it modifies relative links in the HTML to absolute URLs.

    Args:
        html_file_path (str): Path to the local HTML file.
        bucket_name (str): Name of the GCS bucket.
        destination_folder (str, optional): Destination folder in the bucket. Defaults to ''.
        public (bool, optional): Whether to make the uploaded files publicly accessible. Defaults to True.

    Raises:
        FileNotFoundError: If the HTML file or assets folder does not exist.
        Exception: For any other exceptions during the upload process.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if not os.path.isfile(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")

    base_name = os.path.splitext(os.path.basename(html_file_path))[0]
    assets_folder = os.path.join(os.path.dirname(html_file_path), f"{base_name}")

    if not os.path.isdir(assets_folder):
        raise FileNotFoundError(f"Assets folder not found: {assets_folder}")

    with open(html_file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    if destination_folder:
        base_url = f"https://storage.cloud.google.com/{bucket_name}/{destination_folder}/"
    else:
        base_url = f"https://storage.cloud.google.com/{bucket_name}/"

    def make_absolute(url):
        parsed = urlparse(url)
        if not parsed.netloc and not parsed.scheme:
            return urljoin(base_url, url)
        return url

    for tag in soup.find_all(["a", "img", "script", "link"]):
        attr = "href" if tag.name == "a" or tag.name == "link" else "src"
        if tag.has_attr(attr):
            original_url = tag[attr]
            absolute_url = make_absolute(original_url)
            tag[attr] = absolute_url

    modified_html = str(soup)

    html_blob_name = (
        os.path.join(destination_folder, base_name + ".html") if destination_folder else base_name + ".html"
    )
    html_blob = bucket.blob(html_blob_name)
    html_blob.upload_from_string(modified_html, content_type="text/html")
    if public:
        html_blob.make_public()
    print(f"Uploaded HTML file to {html_blob.public_url}")

    def get_mime_type(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    for root, dirs, files in os.walk(assets_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, assets_folder)
            blob_name = (
                os.path.join(destination_folder, base_name, relative_path)
                if destination_folder
                else os.path.join(base_name + "_assets", relative_path)
            )
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path, content_type=get_mime_type(local_path))
            if public:
                blob.make_public()
            print(f"Uploaded {local_path} to {blob.public_url}")

    print("All files uploaded successfully.")


@cli.command()
@click.argument("input_files", type=str, nargs=-1, required=True)
@click.argument("output_file", type=str)
@click.option("--destination-folder", default="", help="Destination folder in the bucket.")
@click.option("--append", multiple=True, help="Text to append to each file's description column.")
@click.option("--public/--private", default=False, help="Make the output file publicly accessible.")
def combine(input_files, output_file, destination_folder, append, public):
    """
    Combines multiple HTML files stored in Google Cloud Storage by merging their rows in order.
    Ensures headers are only included once and allows appending text to the description column for each file.
    """
    client = storage.Client()

    def extract_gcs_info(gcs_url):
        """Extracts bucket name and file path from a GCS public URL."""
        parsed_url = urlparse(gcs_url)
        if not (
            parsed_url.netloc.startswith("storage.googleapis.com")
            or parsed_url.netloc.startswith("storage.cloud.google.com")
        ):
            return None, None

        parts = parsed_url.path.lstrip("/").split("/", 1)
        if len(parts) != 2:
            return None, None

        return parts[0], parts[1]

    def fetch_html(file_source):
        """Fetches HTML content from either a GCS URL or a local file."""
        bucket_name, file_path = extract_gcs_info(file_source)
        if bucket_name and file_path:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            if not blob.exists():
                raise FileNotFoundError(f"File not found in GCS: {file_source}")
            html_content = blob.download_as_text()
        else:
            if not os.path.isfile(file_source):
                raise FileNotFoundError(f"File not found: {file_source}")
            with open(file_source, "r", encoding="utf-8") as f:
                html_content = f.read()

        return BeautifulSoup(html_content, "html.parser")

    soups = []
    tables = []
    for file_path in input_files:
        soup = fetch_html(file_path)
        table = soup.find("table")
        if not table:
            raise ValueError(f"HTML file must contain a <table> element: {file_path}")
        soups.append(soup)
        tables.append(table)

    if not tables:
        raise ValueError("No tables found in the input files.")

    headers = tables[0].find_all("tr")[0]

    all_rows = []
    for i, table in enumerate(tables):
        rows = table.find_all("tr")[1:]
        append_text = append[i] if i < len(append) else ""
        all_rows.append((rows, append_text))

    merged_soup = BeautifulSoup("<html><head></head><body><table></table></body></html>", "html.parser")
    merged_table = merged_soup.find("table")
    merged_table.append(headers)

    def append_description(row, text):
        """Append text to the description column if it exists."""
        if not text:
            return row

        columns = row.find_all("td")
        if len(columns) > 0:
            columns[0].string = f"[ {text} ] | " + (columns[0].string or "")
        return row

    max_rows = max(len(rows) for rows, _ in all_rows)

    for i in range(max_rows):
        for rows, append_text in all_rows:
            if i < len(rows):
                row_copy = BeautifulSoup(str(rows[i]), "html.parser").find("tr")
                merged_table.append(append_description(row_copy, append_text))

    merged_html = str(merged_soup)

    bucket_name = None
    for file_path in input_files:
        bucket_name, _ = extract_gcs_info(file_path)
        if bucket_name:
            break

    if not bucket_name:
        raise ValueError("At least one input file must be a GCS URL to determine the bucket.")

    bucket = client.bucket(bucket_name)
    output_blob_name = os.path.join(destination_folder, output_file) if destination_folder else output_file
    output_blob_name = output_blob_name.replace(".html", f" {uuid.uuid4()}.html")
    output_blob = bucket.blob(output_blob_name)
    output_blob.upload_from_string(merged_html, content_type="text/html")

    if public:
        output_blob.make_public()
        click.echo(f"Combined HTML file uploaded to {output_blob.public_url}")
    else:
        click.echo(
            f"Combined HTML file uploaded to GCS: https://storage.cloud.google.com/{bucket_name}/{output_blob_name}"
        )


if __name__ == "__main__":
    cli()

