import functions_framework


from markupsafe import escape


@functions_framework.http
def validate(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    # the cloud run function will be the first time that our system knows there was a file upload. Once there is a validation event,
    # triggered by the presence of a new file in the bucket with a prefix of "unvalidated/". If validation succeeds, the database is updated and the file is renamed to begin with the validated/ prefix instead. If validation fails, email is triggered with details, the unvalidated/ prefixed file is deleted and no update to database occurs.
    if request_json and "name" in request_json:
        name = request_json["name"]
    elif request_args and "name" in request_args:
        name = request_args["name"]
    else:
        name = "World"
    return f"Hello {escape(name)}!"
