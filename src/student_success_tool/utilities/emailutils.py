""" Utility library for Databricks. Including email functionality.
"""

import smtplib

from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import make_msgid
from google.cloud import storage

SMTP_SERVER = "smtp.mandrillapp.com"
SMTP_PORT = 587  # or 465 for SSL
COMPLETION_SUCCESS_SUBJECT = "Student Success Tool: Inference Results Available."
COMPLETION_SUCCESS_MESSAGE = """\
    Hello!

    Your Datakind Student Success Tool inference run has successfully completed and the results are ready for viewing on the Website.
    """

INFERENCE_KICKOFF_SUBJECT = "Student Success Tool: Inference Run In Progress."
INFERENCE_KICKOFF_MESSAGE = """\
    Hello!

    Your Datakind Student Success Tool inference run has been triggered. Once results have been checked and are ready for viewing, you'll receive another email.
    """


def send_email(
    sender_email,
    receiver_email_list,
    cc_email_list,
    subject,
    body,
    mandrill_username,
    mandrill_password,
):
    # Create the email message
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email_list
    msg["Cc"] = cc_email_list
    msg.set_content(body)
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.login(mandrill_username, mandrill_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise e


def list_blobs_in_ext(bucket_name: str) -> list[str]:
    """Lists all the direct children blobs in the bucket that begin with 'ext/'. Any subfolders or blobs in the subfolders won't be listed.
    You'll get back only the file directly under 'ext/':

        ext/1.txt
    """
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix="ext/", delimiter="/")

    # Note: The call returns a response only when the iterator is consumed.
    res = []
    for blob in blobs:
        res.append(blob.name)
    return res


def send_completion_email(sender_email, receiver_email_list, cc_email_list, username, password):
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        COMPLETION_SUCCESS_SUBJECT,
        COMPLETION_SUCCESS_MESSAGE,
        username,
        password,
    )


def send_inference_kickoff_email(sender_email, receiver_email_list, cc_email_list, username, password):
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        INFERENCE_KICKOFF_SUBJECT,
        INFERENCE_KICKOFF_MESSAGE,
        username,
        password,
    )


def publish_output_files(bucket_name: str, destination_bucket_name: str):
    """Public output files occur once they're approved, at which point, all files staged for publishing, which live in ext/ should get moved to the external bucket which is readable by the webapp."""
    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    blobs_to_move = list_blobs_in_ext(bucket_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)
    if not source_bucket.exists() or not destination_bucket.exists():
        raise ValueError("Unexpected: Storage bucket not found.")
    for elem in blobs_to_move:
        source_blob = source_bucket.blob(elem)
        if not source_blob.exists():
            raise ValueError(elem + ": File not found.")
        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to copy is aborted if the object's
        # generation number does not match your precondition. For a destination
        # object that does not yet exist, set the if_generation_match precondition to 0.
        # If the destination object already exists in your bucket, set instead a
        # generation-match precondition using its generation number.
        # There is also an `if_source_generation_match` parameter, which is not used in this example.
        destination_generation_match_precondition = 0
        new_blob = destination_bucket.blob(elem)
        if new_blob.exists():
            raise ValueError(elem + ": File already exists in destination bucket.")
        # We copy with the same name
        blob_copy = source_bucket.copy_blob(
            source_blob,
            destination_bucket,
            elem,
            if_generation_match=destination_generation_match_precondition,
        )
        # source_blob.delete() # TODO delete the old blob?
