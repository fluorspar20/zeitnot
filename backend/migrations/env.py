"""Alembic environment: the database URL comes from the app settings (ZEITNOT_DATABASE_URL).

A caller can pass a URL explicitly with ``Config.set_main_option("sqlalchemy.url", ...)``
(the loader and the tests do).
"""

from logging.config import fileConfig

from alembic import context

from zeitnot.api.db import make_engine
from zeitnot.api.tables import Base
from zeitnot.config import get_settings

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return config.get_main_option("sqlalchemy.url") or str(get_settings().database_url)


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url())
    with engine.connect() as connection:
        # SQLite cannot alter most column properties in place; batch mode copies the table.
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
