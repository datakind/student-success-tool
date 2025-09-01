"""Utility functions for sending emails in Databricks."""

import smtplib

from email.message import EmailMessage

SMTP_SERVER = "smtp.mandrillapp.com"
# TODO: switch port?
SMTP_PORT = 587  # or 465 for SSL
COMPLETION_SUCCESS_SUBJECT = "Edvise: Inference Results Available"
COMPLETION_SUCCESS_MESSAGE = """\
    Hello!
    
    Your most recent inference results are now available in Edvise. Please log in and download them at your earliest convenience. As a reminder, you should have the StudyID appended file from the NSC SFTP to reconnect our inferences to your students and begin supporting those most in need of intervention.

    We are glad to help consult on anything you need clarification on - should new features have made themselves known or other questions arise, do not hesitate to reach out.

    Best,
    The Edvise Team
    """

INFERENCE_KICKOFF_SUBJECT = "Edvise: Inference Run In Progress"
INFERENCE_KICKOFF_MESSAGE = """\
    Hello!

    Your DataKind Edvise inference run has been successfully initiated. Once the results have been finalized, you will receive a follow-up email with instructions for accessing it.

    Thank you,
    The DataKind team
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
