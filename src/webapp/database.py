import os
import pymysql
import sqlalchemy
import uuid

from google.cloud.sql.connector import Connector
from sqlalchemy.ext.declarative import declarative_base
from contextvars import ContextVar
from sqlalchemy import Column, Uuid, DateTime, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, Session, relationship, mapped_column, Mapped
from sqlalchemy.sql import func

Base = declarative_base()
local_session: ContextVar[Session] = ContextVar("local_session")


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

    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    # If retention unset, the Datakind default is used. File-level retentions overrides
    # this value.
    retention_days: Mapped[int] = mapped_column(nullable=True)
    # A short description or note on this inst.
    description: Mapped[str] = mapped_column(nullable=True)
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
    inst_id: Mapped[str] = mapped_column(
        ForeignKey("inst.id", ondelete="CASCADE"), nullable=False
    )
    inst: Mapped["InstTable"] = relationship(back_populates="accounts")

    name: Mapped[str] = mapped_column(nullable=False)
    email: Mapped[str] = mapped_column(nullable=False, unique=True)
    email_verified: Mapped[bool] = mapped_column(nullable=False)
    access_type: Mapped[str] = mapped_column(nullable=False)
    password_hash: Mapped[str] = mapped_column(nullable=False)
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
    account_id: Mapped[str] = mapped_column(
        ForeignKey("account.id", ondelete="CASCADE"), nullable=False
    )
    account: Mapped["AccountTable"] = relationship(back_populates="account_histories")

    inst_id: Mapped[str] = mapped_column(
        ForeignKey("inst.id", ondelete="CASCADE"), nullable=False
    )
    inst: Mapped["InstTable"] = relationship(back_populates="account_histories")

    action: Mapped[str] = mapped_column(nullable=False)
    resource_id: Mapped[str] = mapped_column(nullable=False)


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
    inst_id: Mapped[str] = mapped_column(
        ForeignKey("inst.id", ondelete="CASCADE"), nullable=False
    )
    inst: Mapped["InstTable"] = relationship(back_populates="files")

    batches: Mapped[set["BatchTable"]] = relationship(
        secondary=association_table, back_populates="files"
    )

    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    # If null, the following is non-deleted.
    deleted: Mapped[bool] = mapped_column(nullable=True)
    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())


# The institution batch table
class BatchTable(Base):
    __tablename__ = "batch"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4())

    # Set the parent foreign key to link to the institution table.
    inst_id: Mapped[str] = mapped_column(
        ForeignKey("inst.id", ondelete="CASCADE"), nullable=False
    )
    inst: Mapped["InstTable"] = relationship(back_populates="batches")

    files: Mapped[set[FileTable]] = relationship(
        secondary=association_table, back_populates="batches"
    )

    name: Mapped[str] = mapped_column(nullable=False, unique=True)
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


# helper function to return SQLAlchemy connection pool
def init_connection_pool(connector: Connector) -> sqlalchemy.engine.Engine:
    # function used to generate database connection
    def getconnauth() -> pymysql.connections.Connection:
        conn = connector.connect(
            os.getenv("INSTANCE_CONNECTION_NAME"),
            "pymysql",
            user=os.getenv("DB_IAM_USER"),
            db=os.getenv("DB_NAME"),
            enable_iam_auth=True,
        )
        return conn

    def getconnpw() -> pymysql.connections.Connection:
        conn = connector.connect(
            os.getenv("INSTANCE_CONNECTION_NAME"),
            "pymysql",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
        )
        return conn

    # create connection pool
    if os.getenv("USE_AUTH") == "True":
        print("[debugging_aaa4] using auth")
        return sqlalchemy.create_engine("mysql+pymysql://", creator=getconnauth)
    print("[debugging_aaa5] using pw")
    return sqlalchemy.create_engine("mysql+pymysql://", creator=getconnpw)


def setup_db():
    # initialize Cloud SQL Python Connector for the GCP databases. Global so
    # shutdown can access it as well.
    print("[debugging_aaa1] Entering setup_db")
    global connector
    if os.getenv("USE_AUTH") == "True":
        connector = Connector(
            ip_type="private", enable_iam_auth=True, refresh_strategy="lazy"
        )
    else:
        connector = Connector(
            ip_type="public", enable_iam_auth=False, refresh_strategy="lazy"
        )
    print("[debugging_aaa2] pass Connector")
    global LocalSession
    # initialize connection pool
    engine = init_connection_pool(connector)
    print("[debugging_aaa3] pass connection pool")
    # Integrating FastAPI with SQL DB
    # create SQLAlchemy ORM session
    LocalSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(engine)
