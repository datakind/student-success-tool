"""Helper functions that may be used across multiple API router subpackages.
"""

from typing import Annotated, Final

# the following needed for python pre 3.11
from strenum import StrEnum
import uuid
import os
import jwt
import re
from fastapi import HTTPException, status, Depends
from pydantic import BaseModel, ConfigDict
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from sqlalchemy.future import select
from urllib.parse import unquote

from .authn import (
    verify_password,
    TokenData,
    oauth2_scheme,
    get_api_key_hash,
    verify_api_key,
    oauth2_apikey_scheme,
)
from .database import get_session, AccountTable, ApiKeyTable
from .config import env_vars


def decode_url_piece(src: str):
    return unquote(src)


class AccessType(StrEnum):
    """Access types available."""

    UNKNOWN = ""
    DATAKINDER = "DATAKINDER"
    MODEL_OWNER = "MODEL_OWNER"
    DATA_OWNER = "DATA_OWNER"
    VIEWER = "VIEWER"


class DataSource(StrEnum):
    """Where the Data was created from."""

    UNNOWN = ""
    PDP_SFTP = "PDP_SFTP"
    MANUAL_UPLOAD = "MANUAL_UPLOAD"


class ApiKeyResponse(BaseModel):
    """The API key object."""

    access_type: AccessType | None = None
    key: str
    inst_id: str | None = None
    allows_enduser: bool | None = None


class ApiKeyRequest(BaseModel):
    """The API key creation object."""

    access_type: AccessType
    inst_id: str | None = None
    allows_enduser: bool | None = None
    valid: bool | None = None
    notes: str | None = None


class UsState(StrEnum):
    """A U.S. state an inst is located in."""

    UNNOWN = ""
    AK = "AK"
    AL = "AL"
    AR = "AR"
    AZ = "AZ"
    CA = "CA"
    CO = "CO"
    CT = "CT"
    DE = "DE"
    FL = "FL"
    GA = "GA"
    HI = "HI"
    IA = "IA"
    ID = "ID"
    IL = "IL"
    IN = "IN"
    KS = "KS"
    KY = "KY"
    LA = "LA"
    MA = "MA"
    MD = "MD"
    ME = "ME"
    MI = "MI"
    MN = "MN"
    MO = "MO"
    MS = "MS"
    MT = "MT"
    NC = "NC"
    ND = "ND"
    NE = "NE"
    NH = "NH"
    NJ = "NJ"
    NM = "NM"
    NV = "NV"
    NY = "NY"
    OH = "OH"
    OK = "OK"
    OR = "OR"
    PA = "PA"
    RI = "RI"
    SC = "SC"
    SD = "SD"
    TN = "TN"
    TX = "TX"
    UT = "UT"
    VA = "VA"
    VT = "VT"
    WA = "WA"
    WI = "WI"
    WV = "WV"
    WY = "WY"


class SchemaType(StrEnum):
    """The schema type a given file adheres to."""

    # If an institution uses UNKNOWN as an allowed schema, it means bypass the validation check.
    UNKNOWN = "UNKNOWN"
    # The standard PDP ARF file schemas
    PDP_COHORT = "PDP_COHORT"
    PDP_COURSE = "PDP_COURSE"
    # The PDP aligned SST schemas
    SST_PDP_COHORT = "SST_PDP_COHORT"
    SST_PDP_COURSE = "SST_PDP_COURSE"
    SST_PDP_FINANCE = "SST_PDP_FINANCE"

    # Schema Types of output files
    SST_OUTPUT = "SST_OUTPUT"
    PNG = "PNG"


PDP_SCHEMA_GROUP: Final = {
    SchemaType.PDP_COHORT,
    SchemaType.PDP_COURSE,
    SchemaType.SST_PDP_FINANCE,
    SchemaType.SST_PDP_COHORT,
    SchemaType.SST_PDP_COURSE,
}


class BaseUser(BaseModel):
    """BaseUser represents an access type. The frontend will include more detailed User info."""

    model_config = ConfigDict(use_enum_values=True)
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers will likely not include a user_id.
    # The actual types of the ids will be UUIDs.
    user_id: str | None = None
    email: str | None = None
    # For Datakinders and people who have not been added to an institution, this is None which means "no inst specified".
    institution: str | None = None
    access_type: AccessType | None = None
    disabled: bool | None = None

    # Constructor
    def __init__(self, usr: str | None, inst: str, access: str, email: str) -> None:
        super().__init__(user_id=usr, institution=inst, access_type=access, email=email)

    def is_datakinder(self) -> bool:
        """Whether a given user is a Datakinder."""
        return self.access_type and self.access_type == AccessType.DATAKINDER

    def is_model_owner(self) -> bool:
        """Whether a given user is a model owner."""
        return self.access_type and self.access_type == AccessType.MODEL_OWNER

    def is_data_owner(self) -> bool:
        """Whether a given user is a data owner."""
        return self.access_type and self.access_type == AccessType.DATA_OWNER

    def is_viewer(self) -> bool:
        """Whether a given user is a viewer."""
        return self.access_type and self.access_type == AccessType.VIEWER

    def has_access_to_inst(self, inst: str) -> bool:
        """Whether a given user has access to a given institution."""
        return self.access_type and (
            self.access_type == AccessType.DATAKINDER or self.institution == inst
        )

    def has_full_data_access(self) -> bool:
        """Datakinders, model_owners, data_owners, all have full data access."""
        return self.access_type and self.access_type in (
            AccessType.DATAKINDER,
            AccessType.MODEL_OWNER,
            AccessType.DATA_OWNER,
        )

    def has_stronger_permissions_than(self, other_access_type: AccessType) -> bool:
        """Check that self has stronger permissions than other."""
        if not self.access_type:
            return False
        if self.access_type == AccessType.DATAKINDER:
            return True
        if self.access_type == AccessType.MODEL_OWNER:
            return other_access_type in (
                AccessType.MODEL_OWNER,
                AccessType.DATA_OWNER,
                AccessType.VIEWER,
            )
        if self.access_type == AccessType.DATA_OWNER:
            return other_access_type in (AccessType.DATA_OWNER, AccessType.VIEWER)
        if self.access_type == AccessType.VIEWER:
            return other_access_type == AccessType.VIEWER
        return False


def get_user(sess: Session, username: str) -> BaseUser:
    if username.startswith("api_key_"):
        api_key_uuid = username.removeprefix("api_key_")
        query_result = sess.execute(
            select(ApiKeyTable).where(
                ApiKeyTable.id == str_to_uuid(api_key_uuid),
            )
        ).all()
        if len(query_result) == 0 or len(query_result) > 1:
            return None
        return BaseUser(
            usr=uuid_to_str(query_result[0][0].id),
            inst=uuid_to_str(query_result[0][0].inst_id),
            access=query_result[0][0].access_type,
            email=username,
        )
    else:
        query_result = sess.execute(
            select(AccountTable).where(
                AccountTable.email == username,
            )
        ).all()
        if len(query_result) == 0 or len(query_result) > 1:
            return None
        return BaseUser(
            usr=uuid_to_str(query_result[0][0].id),
            inst=uuid_to_str(query_result[0][0].inst_id),
            access=query_result[0][0].access_type,
            email=username,
        )


def authenticate_user(
    username: str, password: str, enduser: str | None, sess: Session
) -> BaseUser:
    query_result = sess.execute(
        select(AccountTable).where(
            AccountTable.email == username,
        )
    ).all()
    if len(query_result) == 0 or len(query_result) > 1:
        return False
    if not verify_password(password, query_result[0][0].password):
        return False
    return BaseUser(
        usr=uuid_to_str(query_result[0][0].id),
        inst=uuid_to_str(query_result[0][0].inst_id),
        access=query_result[0][0].access_type,
        email=username,
    )


def authenticate_api_key(api_key_enduser_tuple: str, sess: Session) -> BaseUser:
    (key, inst, enduser) = api_key_enduser_tuple
    query_result = sess.execute(
        select(ApiKeyTable).where(
            ApiKeyTable.inst_id == inst,
        )
    ).all()
    if len(query_result) == 0:
        return False
    for elem in query_result:
        if (
            verify_api_key(key, elem[0].hashed_key_value)
            and elem[0].valid
            and not elem[0].deleted
        ):
            if enduser and query_result[0][0].allows_enduser:
                return get_user(sess, enduser)
            return BaseUser(
                usr=uuid_to_str(elem[0].id),
                inst=uuid_to_str(elem[0].inst_id),
                access=elem[0].access_type,
                email="api_key_" + uuid_to_str(elem[0].id),
            )
    return False


async def get_current_user(
    sess: Annotated[Session, Depends(get_session)],
    token: Annotated[str, Depends(oauth2_scheme)],
    token_from_key: Annotated[str, Depends(oauth2_apikey_scheme)],
) -> BaseUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    usrname = None
    try:
        token_local = None
        if token and token_from_key:
            # Weird to have both, but prioritize non-key token.
            token_local = token
        elif token_from_key:
            token_local = token_from_key
        elif token:
            token_local = token
        else:
            raise credentials_exception
        payload = jwt.decode(
            token_local, env_vars["SECRET_KEY"], algorithms=env_vars["ALGORITHM"]
        )
        usrname = payload.get("sub")
        if usrname is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(sess, username=usrname)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[BaseUser, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def has_access_to_inst_or_err(inst: str, user: BaseUser):
    """Raise error if a given user does not have access to a given institution."""
    if not user.has_access_to_inst(inst):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this institution's resources.",
        )


def has_full_data_access_or_err(user: BaseUser, resource_type: str):
    """Raise error if a given user does not have data access to a given institution."""
    if not user.has_full_data_access():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view " + resource_type + " for this institution.",
        )


def model_owner_and_higher_or_err(user: BaseUser, resource_type: str):
    """Raise error if a given user does not have model ownership or higher."""
    if not user.access_type or user.access_type not in (
        AccessType.MODEL_OWNER,
        AccessType.DATAKINDER,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No permissions for " + resource_type + " for this institution.",
        )


# At this point the value should not be empty as we checked on app startup.
def prepend_env_prefix(name: str) -> str:
    return env_vars["ENV"].lower() + "_" + name


def uuid_to_str(uuid_val: uuid.UUID) -> str:
    if uuid_val is None:
        return ""
    return uuid_val.hex


def str_to_uuid(hex_str: str) -> uuid.UUID:
    return uuid.UUID(hex_str)


def get_external_bucket_name_from_uuid(inst_id: uuid.UUID) -> str:
    return prepend_env_prefix(uuid_to_str(inst_id))


def get_external_bucket_name(inst_id: str) -> str:
    return prepend_env_prefix(inst_id)


def databricksify_inst_name(inst_name: str) -> str:
    """
    Follow DK standardized rules for naming conventions.
    """
    name = inst_name.lower()
    # This needs to be in order from most verbose and encompassing other replacement keys to least.
    dk_replacements = {
        "community technical college": "ctc",
        "community college": "cc",
        "of science and technology": "st",
        "university": "uni",
        "college": "col",
    }

    for old, new in dk_replacements.items():
        name = name.replace(old, new)
    special_char_replacements = {" & ": " ", "&": " ", "-": " "}

    for old, new in special_char_replacements.items():
        name = name.replace(old, new)

    # Databricks uses underscores, so we'll do that here.
    final_name = name.replace(" ", "_")

    # Check to see that no special characters exist
    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, final_name):
        raise ValueError("Unexpected character found in Databricks compatible name.")
    return final_name
