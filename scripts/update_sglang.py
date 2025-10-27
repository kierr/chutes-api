import re
import sys
import uuid
import asyncio
import orjson as json
from loguru import logger
from api.util import notify_deleted
from api.config import settings
from api.image.schemas import Image
from sqlalchemy import select, func, text
from api.database import get_session
from api.chute.schemas import Chute, RollingUpdate
from api.instance.schemas import Instance
from api.graval_worker import handle_rolling_update


async def update_sglang(chute_id: str, image: Image, concurrency: int = 40):
    async with get_session() as session:
        chute = (
            (await session.execute(select(Chute).where(Chute.chute_id == chute_id)))
            .unique()
            .scalar_one_or_none()
        )
        code = chute.code
        if "from_base" in code or "= Image(" in code:
            logger.warning(
                f"Refusing to update chute with image definition: {chute.name=} {chute.chute_id=}"
            )
            return
        code = code.replace(f"chutes/sglang:{chute.image.tag}", f"chutes/{image.name}:{image.tag}")
        if code == chute.code and image.image_id == chute.image_id:
            logger.warning("Code was not changed!")
            return
        if (
            re.search(f" concurrency={chute.concurrency},", code)
            and chute.concurrency < concurrency
        ):
            code = code.replace(
                f" concurrency={chute.concurrency},", f" concurrency={concurrency},"
            )
            logger.info(f"Updated concurrency to {concurrency}")
        else:
            logger.warning("Concurrency unchanged.")
        old_version = chute.version
        chute.code = code
        chute.version = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{image.image_id}:{code}"))
        chute.chutes_version = image.chutes_version
        chute.updated_at = func.now()
        chute.image_id = image.image_id
        logger.success("Updated chute!")

        instances = (
            (await session.execute(select(Instance).where(Instance.chute_id == chute.chute_id)))
            .unique()
            .scalars()
            .all()
        )
        if "affine" in chute.name.lower():
            for inst in instances:
                logger.info(f"Deleting instance: {inst.instance_id=} {inst.miner_hotkey=}")
                await session.delete(inst)
                await asyncio.create_task(notify_deleted(inst))
                await session.execute(
                    text(
                        "UPDATE instance_audit SET deletion_reason = :reason WHERE instance_id = :instance_id"
                    ),
                    {"instance_id": inst.instance_id, "reason": "SGLang/image upgrade"},
                )
        else:
            permitted = {}
            for inst in instances:
                if inst.miner_hotkey not in permitted:
                    permitted[inst.miner_hotkey] = 0
                permitted[inst.miner_hotkey] += 1

            rolling_update = RollingUpdate(
                chute_id=chute.chute_id,
                old_version=old_version,
                new_version=chute.version,
                permitted=permitted,
            )
            session.add(rolling_update)
            logger.info(f"Created rolling update for {chute.name=}")

        await session.commit()
        await session.refresh(chute)
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_updated",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "job_only": not chute.cords,
                    },
                }
            ).decode(),
        )
        if "affine" not in chute.name.lower():
            await handle_rolling_update.kiq(chute.chute_id, chute.version)


async def main(chute_id, concurrency):
    async with get_session() as session:
        image = (
            (
                await session.execute(
                    select(Image).where(Image.image_id == "12911aa5-79e2-5267-abed-46d4c9fd4b7a")
                )
            )
            .unique()
            .scalar_one_or_none()
        )
    await update_sglang(chute_id, image, concurrency)


asyncio.run(main(sys.argv[1], int(sys.argv[2])))
