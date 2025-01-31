"""Main file for the SST API.
"""

import logging
from typing import Any, Annotated
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from datetime import timedelta

from .routers import models, users, data, institutions
from .database import (
    setup_db,
    db_engine,
    Session,
    get_session,
    local_session,
    AccountTable,
    AccountHistoryTable,
)
from .config import env_vars, startup_env_vars

from sqlalchemy.future import select
from sqlalchemy import update
from .utilities import (
    authenticate_user,
    BaseUser,
    get_current_active_user,
    uuid_to_str,
    str_to_uuid,
)
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
    enduser = None
    if len(form_data.scopes) == 1 and form_data.scopes[0]:
        enduser = form_data.scopes[0]
    user = authenticate_user(
        form_data.username, form_data.password, enduser, local_session.get()
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


# Get users that don't have institution specifications. (either datakinders or people who haven't set their institution yet)
@app.get("/non_inst_users", response_model=list[users.UserAccount])
async def read_cross_inst_users(
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
):
    if not current_user.is_datakinder():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Only Datakinders can see non-institution user list",
        )
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(AccountTable).where(
                AccountTable.inst_id == None,
            )
        )
        .all()
    )
    res = []
    if not query_result or len(query_result) == 0:
        return res

    for elem in query_result:
        res.append(
            {
                "user_id": uuid_to_str(elem[0].id),
                "name": elem[0].name,
                "inst_id": uuid_to_str(elem[0].inst_id),
                "access_type": elem[0].access_type,
                "email": elem[0].email,
            }
        )
    return res


# Set access type to datakinder for a list of existing users (by email).
@app.post("/datakinders")
async def set_datakinders(
    emails: list[str],
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> list[str]:
    if not current_user.is_datakinder():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Only Datakinders can set other Datakinders",
        )
    local_session.set(sql_session)
    local_session.get().execute(
        update(AccountTable)
        .where(AccountTable.email.in_(emails))
        .values(access_type="DATAKINDER")
    )
    query_result = (
        local_session.get()
        .execute(
            select(AccountTable).where(
                AccountTable.email.in_(emails),
            )
        )
        .all()
    )

    res = []
    if not query_result:
        return res
    for elem in query_result:
        new_action_record = AccountHistoryTable(
            account_id=str_to_uuid(current_user.user_id),
            action="Set DATAKINDER access type on another user",
            resource_id=elem[0].id,
        )
        res.append(elem[0].email)
        local_session.get().add(new_action_record)
    return res
