import boto3

session = boto3.Session()

credentials = session.get_credentials()

print("===== AWS CREDENTIAL CHECK =====")

if credentials:

    print("Credentials found")

    print(
        "Credential method:",
        credentials.method
    )

    print(
        "Access key prefix:",
        credentials.access_key[:5] + "..."
    )

else:

    print("No credentials found")