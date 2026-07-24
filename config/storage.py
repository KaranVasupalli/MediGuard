"""Storage abstraction: one code path for local files, Azurite, and ADLS Gen2.

The whole point is that no job contains a hardcoded path or a cloud-specific branch.
A job asks for `table_path("corpus")` and gets whatever the current backend is:

  local    ./data/reference/corpus                 (plain folders, fastest to develop)
  azurite  wasbs://mediguard@devstoreaccount1/...  (Azure Storage API, on your laptop)
  adls     abfss://mediguard@<account>.dfs.../...  (real Azure, in the cloud)

Azurite matters because it speaks the REAL Azure Storage API. Proving the Azure code
path works costs nothing locally; discovering it is broken while cloud credits burn is
the expensive way to find out.

Credentials NEVER live in config.yaml — they are read from environment variables, so
nothing secret can be committed. Azurite's key is the exception: it is a public,
documented constant that ships with the emulator and is not a secret.
"""
import os

from config.spark_config import load_config

# Azurite's well-known development credentials — published by Microsoft, identical on
# every machine, and deliberately NOT a secret.
AZURITE_ACCOUNT = "devstoreaccount1"
AZURITE_KEY = ("Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==")


def backend() -> str:
    cfg = load_config()
    return cfg.get("storage", {}).get("backend", "local")


def container() -> str:
    cfg = load_config()
    return cfg.get("storage", {}).get("container", "mediguard")


def table_path(name: str) -> str:
    """Where a named table lives under the current backend."""
    cfg = load_config()
    st = cfg.get("storage", {})
    b = st.get("backend", "local")

    if b == "local":
        return f"{cfg['paths']['reference']}/{name}"

    if b == "azurite":
        return f"wasbs://{container()}@{AZURITE_ACCOUNT}.blob.core.windows.net/{name}"

    if b == "adls":
        account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        if not account:
            raise RuntimeError(
                "AZURE_STORAGE_ACCOUNT is not set. Export it before using the adls "
                "backend; it must never be written into config.yaml.")
        return f"abfss://{container()}@{account}.dfs.core.windows.net/{name}"

    raise ValueError(f"unknown storage backend {b!r}")


def spark_storage_options() -> dict:
    """Spark config needed to reach the current backend."""
    b = backend()

    if b == "local":
        return {}

    if b == "azurite":
        # Point the Azure connector at the local emulator instead of the cloud.
        host = os.environ.get("AZURITE_HOST", "127.0.0.1")
        return {
            f"fs.azure.account.key.{AZURITE_ACCOUNT}.blob.core.windows.net": AZURITE_KEY,
            "fs.azure.storage.emulator.account.name": AZURITE_ACCOUNT,
            f"fs.azure.account.auth.type.{AZURITE_ACCOUNT}.blob.core.windows.net": "SharedKey",
            "spark.hadoop.fs.azure.test.emulator": "true",
            "spark.hadoop.fs.azure.storage.emulator.account.name": AZURITE_ACCOUNT,
            "_azurite_host": host,      # informational; used by the runner's log line
        }

    if b == "adls":
        account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        key = os.environ.get("AZURE_STORAGE_KEY", "")
        if not account or not key:
            raise RuntimeError(
                "Set AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY in the environment. "
                "Never put them in config.yaml or commit them.")
        return {
            f"fs.azure.account.key.{account}.dfs.core.windows.net": key,
            f"fs.azure.account.auth.type.{account}.dfs.core.windows.net": "SharedKey",
        }

    return {}


def deltalake_storage_options() -> dict:
    """Equivalent options for the delta-rs (non-Spark) writer used locally."""
    b = backend()
    if b == "azurite":
        return {"AZURE_STORAGE_ACCOUNT_NAME": AZURITE_ACCOUNT,
                "AZURE_STORAGE_ACCOUNT_KEY": AZURITE_KEY,
                "AZURE_STORAGE_USE_EMULATOR": "true"}
    if b == "adls":
        return {"AZURE_STORAGE_ACCOUNT_NAME": os.environ.get("AZURE_STORAGE_ACCOUNT", ""),
                "AZURE_STORAGE_ACCOUNT_KEY": os.environ.get("AZURE_STORAGE_KEY", "")}
    return {}


def describe() -> str:
    b = backend()
    return {
        "local": "local filesystem (./data)",
        "azurite": "Azurite — Azure Storage API, running locally in Docker",
        "adls": f"Azure Data Lake Storage Gen2 (account: "
                f"{os.environ.get('AZURE_STORAGE_ACCOUNT', 'NOT SET')})",
    }.get(b, b)
