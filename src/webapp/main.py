"""Main file for the SST API.
"""

import logging
from typing import Any, Annotated
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from datetime import timedelta

from .routers import models, users, data, institutions
from .database import setup_db, db_engine, Session, get_session, local_session
from .config import env_vars, startup_env_vars

from .utilities import authenticate_user
from .authn import (
    Token,
    create_access_token,
)

# Set the logging
logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI(
    servers=[
        # TODO: placeholders
        {"url": "https://stag.example.com", "description": "Staging environment"},
        {"url": "https://prod.example.com", "description": "Production environment"},
    ],
    root_path="/api/v1",
)


"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

app.include_router(institutions.router)
app.include_router(models.router)
app.include_router(users.router)
app.include_router(data.router)


@app.on_event("startup")
def on_startup():
    print("Starting up app...")
    startup_env_vars()
    setup_db(env_vars["ENV"])


# On shutdown, we have to cleanup the GCP database connections
@app.on_event("shutdown")
async def shutdown_event():
    print("Performing shutdown tasks...")
    await db_engine.dispose()


# The following root paths don't have pre-authn.
@app.get("/")
def read_root() -> Any:
    """Returns the index.html file."""
    return FileResponse("src/webapp/index.html")


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Token:
    local_session.set(sql_session)
    user = authenticate_user(
        form_data.username, form_data.password, local_session.get()
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(
        minutes=int(env_vars["ACCESS_TOKEN_EXPIRE_MINUTES"])
    )
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


# Get users that are cross-institution
"""
@app.get("/users", response_model=UserAccount)
async def read_users_me(
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
):
    return {}
    
    return {
        "user_id": uuid,
        "name": user_account_request.name,
        "inst_id": inst_id,
        "access_type": user_account_request.access_type,
        "email": user_account_request.email,
        "username": user_account_request.username,
    }
    """
