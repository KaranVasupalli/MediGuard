from azure.storage.blob import BlobServiceClient

CONN = ("DefaultEndpointsProtocol=http;"
        "AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;")

svc = BlobServiceClient.from_connection_string(CONN, api_version="2023-11-03")

try:
    svc.create_container("mediguard")
    print("container created")
except Exception as e:
    print("already exists or:", type(e).__name__)