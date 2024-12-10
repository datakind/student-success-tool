import os
import pymysql
import sqlalchemy
import uuid

from sqlalchemy.ext.declarative import declarative_base
from contextvars import ContextVar
from sqlalchemy import Column, Uuid, DateTime, ForeignKey, Table, String
from sqlalchemy.orm import sessionmaker, Session, relationship, mapped_column, Mapped
from sqlalchemy.sql import func

Base = declarative_base()
LocalSession = None
local_session: ContextVar[Session] = ContextVar("local_session")
db_engine = None

# GCP MYSQL will throw an error if we don't specify the length for any varchar
# fields. So we can't use mapped_column in string cases.
VAR_CHAR_LENGTH = 30


# The institution overview table that maps ids to names. The parent table to
# all other ables except for AccountHistory.
class InstTable(Base):
    __tablename__ = "inst"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4())

    # Linked children tables.
    accounts: Mapped[set["AccountTable"]] = relationship(back_populates="inst")
    files: Mapped[set["FileTable"]] = relationship(back_populates="inst")
    batches: Mapped[set["BatchTable"]] = relationship(back_populates="inst")
    account_histories: Mapped[list["AccountHistoryTable"]] = relationship(
        back_populates="inst"
    )

    name = Column(String(VAR_CHAR_LENGTH), nullable=False, unique=True)
    # If retention unset, the Datakind default is used. File-level retentions overrides
    # this value.
    retention_days: Mapped[int] = mapped_column(nullable=True)
    # A short description or note on this inst.
    description = Column(String(VAR_CHAR_LENGTH))
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


# The user accounts table
class AccountTable(Base):
    __tablename__ = "account"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4())

    # Set account histories to be children
    account_histories: Mapped[list["AccountHistoryTable"]] = relationship(
        back_populates="account"
    )

    # Set the foreign key to link to the institution table.
    inst_id = Column(
        String(VAR_CHAR_LENGTH),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="accounts")

    name = Column(String(VAR_CHAR_LENGTH), nullable=False)
    email = Column(String(VAR_CHAR_LENGTH), nullable=False, unique=True)
    email_verified: Mapped[bool] = mapped_column(nullable=False)
    access_type = Column(String(VAR_CHAR_LENGTH), nullable=False)
    password_hash = Column(String(VAR_CHAR_LENGTH), nullable=False)
    time_created = mapped_column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    inst = relationship("InstTable", backref="account")


# The user history table
class AccountHistoryTable(Base):
    __tablename__ = "account_history"
    time_occurred = Column(
        DateTime(timezone=True), server_default=func.now(), primary_key=True
    )

    # Set the parent foreign key to link to the accounts table.
    account_id = Column(
        String(VAR_CHAR_LENGTH),
        ForeignKey("account.id", ondelete="CASCADE"),
        nullable=False,
    )
    account: Mapped["AccountTable"] = relationship(back_populates="account_histories")

    inst_id = Column(
        String(VAR_CHAR_LENGTH),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="account_histories")

    action = Column(String(VAR_CHAR_LENGTH), nullable=False)
    resource_id = Column(String(VAR_CHAR_LENGTH), nullable=False)


# An intermediary association table allows bi-directional many-to-many between files and batches.
association_table = Table(
    "association_table",
    Base.metadata,
    Column("file_val", ForeignKey("file.id"), primary_key=True),
    Column("batch_val", ForeignKey("batch.id"), primary_key=True),
)


# The institution file table
class FileTable(Base):
    __tablename__ = "file"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4())

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        String(VAR_CHAR_LENGTH),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="files")

    batches: Mapped[set["BatchTable"]] = relationship(
        secondary=association_table, back_populates="files"
    )

    name = Column(String(VAR_CHAR_LENGTH), nullable=False, unique=True)
    # If null, the following is non-deleted.
    deleted: Mapped[bool] = mapped_column(nullable=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


# The institution batch table
class BatchTable(Base):
    __tablename__ = "batch"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4())

    # Set the parent foreign key to link to the institution table.
    inst_id = Column(
        String(VAR_CHAR_LENGTH),
        ForeignKey("inst.id", ondelete="CASCADE"),
        nullable=False,
    )
    inst: Mapped["InstTable"] = relationship(back_populates="batches")

    files: Mapped[set[FileTable]] = relationship(
        secondary=association_table, back_populates="batches"
    )

    name = Column(String(VAR_CHAR_LENGTH), nullable=False, unique=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


def get_session():
    sess: Session = LocalSession()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
    finally:
        sess.close()


# The following functions from the GCP open source sample code. xxxxxxxx


def connect_tcp_socket(
    engine_args: dict[str, str], connect_args: dict[str, str]
) -> sqlalchemy.engine.base.Engine:
    """Initializes a TCP connection pool for a Cloud SQL instance of MySQL."""
    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # mysql+pymysql://<db_user>:<db_pass>@<db_host>:<db_port>/<db_name>
        sqlalchemy.engine.url.URL.create(
            drivername="mysql+pymysql",
            username=engine_args["DB_USER"],
            password=engine_args["DB_PASS"],
            host=engine_args["INSTANCE_HOST"],
            port=engine_args["DB_PORT"],
            database=engine_args["DB_NAME"],
        ),
        # [END cloud_sql_mysql_sqlalchemy_connect_tcp]
        connect_args=connect_args,
        # [START cloud_sql_mysql_sqlalchemy_connect_tcp]
        # [START_EXCLUDE]
        # [START cloud_sql_mysql_sqlalchemy_limit]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # [END cloud_sql_mysql_sqlalchemy_limit]
        # [START cloud_sql_mysql_sqlalchemy_backoff]
        # SQLAlchemy automatically uses delays between failed connection attempts,
        # but provides no arguments for configuration.
        # [END cloud_sql_mysql_sqlalchemy_backoff]
        # [START cloud_sql_mysql_sqlalchemy_timeout]
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # [END cloud_sql_mysql_sqlalchemy_timeout]
        # [START cloud_sql_mysql_sqlalchemy_lifetime]
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        # [END cloud_sql_mysql_sqlalchemy_lifetime]
        # [END_EXCLUDE]
    )
    return pool


# helper function to return SQLAlchemy connection pool
def init_connection_pool() -> (
    sqlalchemy.engine.Engine
):  # should this be sqlalchemy.engine.base.Engine
    # The INSTANCE_HOST is the private IP of CLoudSQL instance e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
    engine_args = {
        "INSTANCE_HOST": os.environ.get("INSTANCE_HOST"),
        "DB_USER": os.environ.get("DB_USER"),
        "DB_PASS": os.environ.get("DB_PASS"),
        "DB_NAME": os.environ.get("DB_NAME"),
        "DB_PORT": os.environ.get("DB_PORT"),
    }
    for elem_name, value in engine_args.items():
        if not value:
            raise ValueError("Missing " + elem_name + " value missing. Required.")
    # For deployments that connect directly to a Cloud SQL instance without
    # using the Cloud SQL Proxy, configuring SSL certificates will ensure the
    # connection is encrypted.
    # root cert: e.g. '/path/to/server-ca.pem'
    # cert: e.g. '/path/to/client-cert.pem'
    # key: e.g. '/path/to/client-key.pem'
    ssl_env_vars = {
        "DB_ROOT_CERT": os.environ.get("DB_ROOT_CERT"),
        "DB_CERT": os.environ.get("DB_CERT"),
        "DB_KEY": os.environ.get("DB_KEY"),
    }

    for elem_name, value in ssl_env_vars.items():
        if not value:
            raise ValueError(
                "Missing " + elem_name + " value missing. Required for SSL connection."
            )
    ssl_args = {
        "ssl_ca": ssl_env_vars["DB_ROOT_CERT"],
        "ssl_cert": ssl_env_vars["DB_CERT"],
        "ssl_key": ssl_env_vars["DB_KEY"],
    }
    return connect_tcp_socket(engine_args, ssl_args)


def setup_db():
    # initialize connection pool
    global db_engine
    db_engine = init_connection_pool()
    # Integrating FastAPI with SQL DB
    # create SQLAlchemy ORM session
    global LocalSession
    LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    Base.metadata.create_all(db_engine)
