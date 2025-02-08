import jwt

from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    APIKeyHeader,
    APIKeyQuery,
)
from pydantic import BaseModel
from datetime import timedelta, datetime, timezone
from .config import env_vars
from typing import Annotated
from fastapi import Depends, HTTPException, status


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
)


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


def check_creds(username: str, password: str):
    if username == env_vars["USERNAME"] and password == env_vars["PASSWORD"]:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Creds for worker job not correct",
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=env_vars["ACCESS_TOKEN_EXPIRE_MINUTES"]
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, env_vars["SECRET_KEY"], algorithm=env_vars["ALGORITHM"]
    )
    return encoded_jwt


async def get_current_username(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    username = ""
    try:
        payload = jwt.decode(
            token, env_vars["SECRET_KEY"], algorithms=env_vars["ALGORITHM"]
        )
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    if token_data.username != env_vars["USERNAME"]:
        raise credentials_exception
    return username
