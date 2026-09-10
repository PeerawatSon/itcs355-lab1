"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter


class AzureAdapter(CloudAdapter):
    def _parse_blob_uri(self, uri: str) -> tuple[str, str, str]:
        """Parse https://<account>.blob.core.windows.net/<container>/<prefix>"""
        p = urlparse(uri)
        account = p.netloc.split(".")[0]
        parts = p.path.lstrip("/").split("/", 1)
        container = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return account, container, prefix

    def upload(self, local_path: str, key: str) -> str:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account, container, prefix = self._parse_blob_uri(self.cfg.blob_uri)
        blob_name = f"{prefix.rstrip('/')}/{key.lstrip('/')}" if prefix else key
        account_url = f"https://{account}.blob.core.windows.net"

        service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)

        with open(local_path, "rb") as fh:
            blob_client.upload_blob(fh, overwrite=True)

        return f"{account_url}/{container}/{blob_name}"

    def download(self, uri: str, local_path: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account, container, blob_name = self._parse_blob_uri(uri)
        account_url = f"https://{account}.blob.core.windows.net"

        service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)

        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            download_stream = blob_client.download_blob()
            download_stream.readinto(fh)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry
        server = registry.split("/")[0]
        acr_name = server.split(".")[0]

        # 1. Login to Azure Container Registry
        subprocess.run(["az", "acr", "login", "--name", acr_name], check=True)

        # 2. Tag local image for remote ACR
        tag_suffix = local_tag.split(":")[-1] if ":" in local_tag else "latest"
        remote_tag = f"{registry}:{tag_suffix}"
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)

        # 3. Push to ACR
        subprocess.run(["docker", "push", remote_tag], check=True)

        # 4. Get digest reference (registry.azurecr.io/repo@sha256:...)
        res = subprocess.run(
            ["docker", "inspect", "--format={{range .RepoDigests}}{{.}}\n{{end}}", remote_tag],
            capture_output=True,
            text=True,
            check=True,
        )
        digests = [d.strip() for d in res.stdout.splitlines() if d.strip()]
        matching = [d for d in digests if d.startswith(registry)]
        if matching:
            return matching[0]

        repo = registry.split("/", 1)[1] if "/" in registry else local_tag.split(":")[0]
        az_res = subprocess.run(
            ["az", "acr", "repository", "show", "--name", acr_name, "--image", f"{repo}:{tag_suffix}", "--query", "digest", "-o", "tsv"],
            capture_output=True,
            text=True,
            check=True,
        )
        digest = az_res.stdout.strip()
        return f"{registry}@{digest}"

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
