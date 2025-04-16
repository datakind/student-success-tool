"""Utility functions for sending emails in Databricks."""

import smtplib

from email.message import EmailMessage

SMTP_SERVER = "smtp.mandrillapp.com"
# TODO: switch port?
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
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    subject: str,
    body: str,
    mandrill_username: str,
    mandrill_password: str,
) -> None:
    """Send email.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      subject: Subject of email.
      body: Body of email.
      mandrill_username: Mandrill username for the SMTP server.
      mandrill_password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
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


def send_inference_completion_email(
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    username: str,
    password: str,
) -> None:
    """Send email with completion of inference run message.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      username: Mandrill username for the SMTP server.
      password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        COMPLETION_SUCCESS_SUBJECT,
        COMPLETION_SUCCESS_MESSAGE,
        username,
        password,
    )


def send_inference_kickoff_email(
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    username: str,
    password: str,
) -> None:
    """Send email with kickoff of inference run message.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      username: Mandrill username for the SMTP server.
      password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        INFERENCE_KICKOFF_SUBJECT,
        INFERENCE_KICKOFF_MESSAGE,
        username,
        password,
    )
