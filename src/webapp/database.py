import os
import sqlalchemy
import uuid
import datetime

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict, MutableList
from contextvars import ContextVar
from sqlalchemy import (
    Column,
    Uuid,
    DateTime,
    ForeignKey,
    Table,
    String,
    UniqueConstraint,
    Text,
    JSON,
    Integer,
)
from typing import Set, List
from sqlalchemy.orm import sessionmaker, Session, relationship, mapped_column, Mapped
from sqlalchemy.sql import func
from sqlalchemy.pool import StaticPool
from .config import env_vars, engine_vars, ssl_env_vars, setup_database_vars
from .authn import get_password_hash

Base = declarative_base()
LocalSession = None
local_session: ContextVar[Session] = ContextVar("local_session")
db_engine = None

# GCP MYSQL will throw an error if we don't specify the length for any varchar
# fields. So we can't use mapped_column in string cases.
VAR_CHAR_LENGTH = 36
VAR_CHAR_LONGER_LENGTH = 1000
# Constants for the local env
LOCAL_INST_UUID = uuid.UUID("14c81c50-935e-4151-8561-c2fc3bdabc0f")
LOCAL_USER_UUID = uuid.UUID("f21a3e53-c967-404e-91fd-f657cb922c39")
LOCAL_USER_EMAIL = "tester@datakind.org"
HASHED_PASSWORD = "tester_password"
DATETIME_TESTING = datetime.datetime(2024, 12, 26, 19, 37, 59, 753357)


def init_db(env: str):
    # add some sample users to the database for development utility.
    if env not in ("LOCAL", "DEV"):
        # this init db setup only applies for Local and Dev.
        return
    try:
        with sqlalchemy.orm.Session(db_engine) as session:
            session.merge(
                AccountTable(
                    id=LOCAL_USER_UUID,
                    inst_id=None,
                    name="Tester S",
                    email=(
                        LOCAL_USER_EMAIL
                        if env == "LOCAL"
                        else os.getenv("DEV_INIT_DB_USER")
                    ),
                    email_verified_at=None,
                    password=(
                        get_password_hash(
                            HASHED_PASSWORD
                            if env == "LOCAL"
                            else os.getenv("DEV_INIT_DB_PASSWORD")
                        )
                    ),
                    access_type="DATAKINDER",
                    created_at=DATETIME_TESTING,
                    updated_at=DATETIME_TESTING,
                )
            )
            session.merge(
                InstTable(
                    id=LOCAL_INST_UUID,
                    name="Foobar University",
                    created_at=DATETIME_TESTING,
                    updated_at=DATETIME_TESTING,
                )
            )
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# The institution overview table that maps ids to names. The parent table to
# all other ables except for AccountHistory.
class InstTable(Base):
    __tablename__ = "inst"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Linked children tables.
    accounts: Mapped[Set["AccountTable"]] = relationship(back_populates="inst")
    files: Mapped[Set["FileTable"]] = relationship(back_populates="inst")
    batches: Mapped[Set["BatchTable"]] = relationship(back_populates="inst")
    account_histories: Mapped[List["AccountHistoryTable"]] = relationship(
        back_populates="inst"
    )
    # models: Mapped[List["ModelTable"]] = relationship(
    #    back_populates="inst"
    # )

    name = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False, unique=True)
    # If retention unset, the Datakind default is used. File-level retentions overrides
    # this value.
    retention_days: Mapped[int] = mapped_column(nullable=True)
    # The emails for which self sign up will be allowed for this institution and will automatically be assigned to this institution.
    # The dict structure is {email: AccessType string}
    allowed_emails = Column(MutableDict.as_mutable(JSON))
    # Schemas that are allowed for validation.
    schemas = Column(MutableList.as_mutable(JSON))
    state = Column(String(VAR_CHAR_LENGTH), nullable=True)
    # Only populated for PDP schools. Uncomment once logic available.
    # pdp_id: Mapped[int] = mapped_column(nullable=True)
    # A short description or note on this inst.
    description = Column(String(VAR_CHAR_LONGER_LENGTH))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Within the institutions, the set of name + state should be unique
    __table_args__ = (UniqueConstraint("name", "state", name="inst_name_state_uc"),)


# NOTE: only users created by the frontend are accessible through the fronted. Users created by API calls can only directly call API calls. Frontend will not work.
# The user accounts table
class AccountTable(Base):
    __tablename__ = "users"  # Name to be compliant with Laravel.
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Set account histories to be children
    account_histories: Mapped[List["AccountHistoryTable"]] = relationship(
        back_populates="account"
    )

    # Set the foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=True,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="accounts")

    name = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    email = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False, unique=True)
    google_id = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=True)
    azure_id = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=True)

    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    password = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    two_factor_secret = Column(Text, nullable=True)
    two_factor_recovery_codes = Column(Text, nullable=True)
    two_factor_confirmed_at = Column(DateTime(timezone=True), nullable=True)

    remember_token = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=True)
    # Required for team integration with laravel
    current_team_id = Column(Uuid(as_uuid=True), nullable=True)
    access_type = Column(String(VAR_CHAR_LENGTH), nullable=True)
    profile_photo_path = Column(String(VAR_CHAR_LENGTH), nullable=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    inst = relationship("InstTable", backref="account")


# The user history table
class AccountHistoryTable(Base):
    __tablename__ = "account_history"
    id = Column(Integer, primary_key=True)  # Auto-increment should be default
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Set the parent foreign key to link to the users table.
    account_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    account: Mapped["AccountTable"] = relationship(back_populates="account_histories")

    # This field is nullable if the action was taken by a Datakinder.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=True,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="account_histories")

    action = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    resource_id = Column(Uuid(as_uuid=True), nullable=False)


# An intermediary association table allows bi-directional many-to-many between files and batches.
association_table = Table(
    "file_batch_association_table",
    Base.metadata,
    Column("file_val", Uuid(as_uuid=True), ForeignKey("file.id"), primary_key=True),
    Column("batch_val", Uuid(as_uuid=True), ForeignKey("batch.id"), primary_key=True),
)


# The institution file table
class FileTable(Base):
    __tablename__ = "file"
    name = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batches: Mapped[Set["BatchTable"]] = relationship(
        secondary=association_table, back_populates="files"
    )
    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="files")
    # The size to the nearest mb.
    # size_mb: Mapped[int] = mapped_column(nullable=False)
    # A short description or note on this inst.
    description = Column(String(VAR_CHAR_LONGER_LENGTH))
    # Who uploaded the file. For SST generated files, this field would be null.
    uploader = Column(Uuid(as_uuid=True), nullable=True)
    # Can be PDP_SFTP, MANUAL_UPLOAD etc. May be empty for generated files.
    source = Column(String(VAR_CHAR_LENGTH), nullable=True)
    # The schema type(s) of this file.
    schemas = Column(MutableList.as_mutable(JSON), nullable=False)
    # If null, the following is non-deleted.
    # The deleted field indicates whether there is a pending deletion request on the data.
    # The data may stil be available to Datakind debug role in a soft-delete state but for all
    # intents and purposes is no longer accessible by the app.
    deleted: Mapped[bool] = mapped_column(nullable=True)
    # When the deletion request was made
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    retention_days: Mapped[int] = mapped_column(nullable=True)
    # Whether the file was generated by SST. (e.g. was it input or output)
    sst_generated: Mapped[bool] = mapped_column(nullable=False)
    # Whether the file was or approved (in the case of output).
    valid: Mapped[bool] = mapped_column(nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Within a given institution, there should be no duplicated file names.
    __table_args__ = (UniqueConstraint("name", "inst_id", name="file_name_inst_uc"),)


# The institution batch table
class BatchTable(Base):
    __tablename__ = "batch"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="batches")

    files: Mapped[Set[FileTable]] = relationship(
        secondary=association_table, back_populates="batches"
    )

    name = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    # A short description or note on this inst.
    description = Column(String(VAR_CHAR_LONGER_LENGTH))
    creator = Column(Uuid(as_uuid=True))
    # If null, the following is non-deleted.
    deleted: Mapped[bool] = mapped_column(nullable=True)
    # If true, the batch is ready for use.
    completed: Mapped[bool] = mapped_column(nullable=True)
    # The time the deletion request was set.
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # Within a given institution, there should be no duplicated batch names.
    __table_args__ = (UniqueConstraint("name", "inst_id", name="batch_name_inst_uc"),)


"""
class ModelTable(Base):
    __tablename__ = "model"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="models")

    name = Column(String(VAR_CHAR_LONGER_LENGTH), nullable=False)
    # A short description or note on this inst.
    description = Column(String(VAR_CHAR_LONGER_LENGTH))
    # What configuration of schemas are allowed (e.g. 1 PDP Course + 1 PDP Cohort)
    schema_configs = Column(MutableList.as_mutable(JSON))
    creator = Column(Uuid(as_uuid=True))
    # If null, the following is non-deleted.
    deleted: Mapped[bool] = mapped_column(nullable=True)
    # If true, the model is ready for use.
    active: Mapped[bool] = mapped_column(nullable=True)
    # The time the deletion request was set.
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # Within a given institution, there should be no duplicated model names.
    __table_args__ = (UniqueConstraint("name", "inst_id", name="model_name_inst_uc"),)

class ModelRunsTable(Base):
    __tablename__ = "model_runs"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="batches")

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="batches")

"""
"""
def get_one_record(sess_context_var: ContextVar, sess: Session, select_query: ) -> Any:
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(select(BatchTable).where(BatchTable.name == req.name))
        .all()
    )
    if len(query_result) == 0:
        # If the institution does not exist create it and create a storage bucket for it.
        # TODO Check presence of storage bucket, if it does not exist, create it.
        local_session.get().add(
            BatchTable(
                name=req.name,
                inst_id=inst_id,
                description=req.description,
                creator=current_user.user_id,
            )
        )
        local_session.get().commit()
        query_result = (
            local_session.get()
            .execute(select(BatchTable).where(BatchTable.name == req.name))
            .all()
        )
        if not query_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the batch creation failed.",
            )
        elif len(query_result) > 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the batch created duplicate entries.",
            )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch duplicates found.",
        )
    return query_result[0][0]
"""


def get_session():
    sess: Session = LocalSession()
    try:
        yield sess
        sess.commit()
    except Exception as e:
        sess.rollback()
        raise e
    finally:
        sess.close()


def init_connection_pool_local() -> sqlalchemy.engine.base.Engine:
    # Creates a local sqlite db for local env testing
    return sqlalchemy.create_engine(
        "sqlite://",
        echo=True,
        echo_pool="debug",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


# The following functions are heavily inspired from the GCP open source sample code.


def connect_tcp_socket(
    engine_args: dict[str, str], connect_args: dict[str, str]
) -> sqlalchemy.engine.base.Engine:
    """Initializes a TCP connection pool for a Cloud SQL instance of MySQL."""
    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # mysql+pymysql://<db_user>:<db_pass>@<db_host>:<db_port>/<db_name>
        # https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/cloud-sql/postgres/sqlalchemy/connect_tcp.py
        sqlalchemy.engine.url.URL.create(
            drivername="mysql+pymysql",
            username=engine_args["DB_USER"],
            password=engine_args["DB_PASS"],
            host=engine_args["INSTANCE_HOST"],
            port=engine_args["DB_PORT"],
            database=engine_args["DB_NAME"],
        ),
        connect_args=connect_args,
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # SQLAlchemy automatically uses delays between failed connection attempts,
        # but provides no arguments for configuration.
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
    )
    return pool


# helper function to return SQLAlchemy connection pool
def init_connection_pool() -> (
    sqlalchemy.engine.Engine
):  # should this be sqlalchemy.engine.base.Engine
    setup_database_vars()
    # Set up ssl context for the connection args.
    ssl_args = {
        "ssl_ca": ssl_env_vars["DB_ROOT_CERT"],
        "ssl_cert": ssl_env_vars["DB_CERT"],
        "ssl_key": ssl_env_vars["DB_KEY"],
    }
    return connect_tcp_socket(engine_vars, ssl_args)


def setup_db(env: str):
    # initialize connection pool
    global db_engine
    if env == "LOCAL":
        db_engine = init_connection_pool_local()
    else:
        db_engine = init_connection_pool()
    # Integrating FastAPI with SQL DB
    # create SQLAlchemy ORM session
    global LocalSession
    LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    # TODO: instead of create_all, check if exists and only create all if no exist
    # https://stackoverflow.com/questions/33053241/sqlalchemy-if-table-does-not-exist
    Base.metadata.create_all(db_engine)
    if env in ("LOCAL", "DEV"):
        # Creates a fake user in the local db
        init_db(env)
