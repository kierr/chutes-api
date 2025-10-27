import api.database.orms  # noqa
import sys
import asyncio
from api.database import get_session
from sqlalchemy import select
from api.instance.schemas import Instance
import api.miner_client as miner_client


async def get_logs(instance_id: str):
    async with get_session() as session:
        instance = (
            (await session.execute(select(Instance).where(Instance.instance_id == instance_id)))
            .unique()
            .scalar_one_or_none()
        )
        log_port = next(p for p in instance.port_mappings if p["internal_port"] == 8001)[
            "external_port"
        ]
        # async with miner_client.get(instance.miner_hotkey, f"http://{instance.host}:{log_port}/logs/read/current", purpose="chutes") as resp:
        async with miner_client.get(
            instance.miner_hotkey,
            f"http://{instance.host}:{log_port}/logs/stream?backfill=1000",
            timeout=0,
            purpose="chutes",
        ) as resp:
            async for chunk in resp.content:
                cont = chunk.decode().strip()
                if cont not in ("", "."):
                    print(cont)


asyncio.run(get_logs(sys.argv[1]))
