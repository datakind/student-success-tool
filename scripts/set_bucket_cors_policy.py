import sys
from google.cloud import storage
from typing import List


def cors_configuration(bucket_name: str, origin: str, methods: List[str] = ['PUT', 'POST']) -> storage.Bucket:
    """Set a bucket's CORS policies configuration."""

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    bucket.cors = [
        {
            "origin": [origin],
            "responseHeader": [
                "Content-Type",
                "x-goog-resumable"],
            "method": methods,
            "maxAgeSeconds": 3600
        }
    ]
    bucket.patch()

    return bucket


def main():
    bucket_name, origin = sys.argv[1], sys.argv[2]
    bucket = cors_configuration(bucket_name, origin).cors
    print(f"Set CORS policies for bucket {bucket.name} is {bucket.cors}")



if __name__ == "__main__":
    main()
