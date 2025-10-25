"""
Application-wide settings.
"""

import os
from typing import Optional
from fiber import Keypair
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    subtensor: str = os.getenv("SUBTENSOR_ADDRESS", "wss://entrypoint-finney.opentensor.ai:443")
    netuid: int = os.getenv("NETUID", "64")
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Postgres configuration (read-write and read-only replica).
    db_rw_url: str = os.getenv("DB_RW_URL")
    db_ro_url: str = os.getenv("DB_RO_URL")

    # Database settings.
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "16"))
    db_overflow: int = int(os.getenv("DB_OVERFLOW", "3"))

    validator_ss58: Optional[str] = os.getenv("VALIDATOR_SS58")
    validator_keypair: Optional[Keypair] = (
        Keypair.create_from_seed(os.environ["VALIDATOR_SEED"])
        if os.getenv("VALIDATOR_SEED")
        else None
    )

    subtensor_network: Optional[str] = Field(None, alias="SUBTENSOR_NETWORK")
    subtensor_address: Optional[str] = Field(None, alias="SUBTENSOR_ADDRESS")


settings = Settings()
