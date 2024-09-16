from os import getenv

from fastapi import FastAPI, File, HTTPException

from typing import Annotated

import redis.asyncio as redis
from redis.commands.search.query import Query

import numpy as np
import tensorflow as tf

from hashlib import sha256

#REDIS_HOST = getenv( "REDIS_HOST" )
#REDIS_PORT = getenv( "REDIS_PORT" )
REDIS_HOST = "localhost"
REDIS_PORT = "6379"


app = FastAPI()

r = redis.Redis( host = REDIS_HOST, port = REDIS_PORT )


@app.post( "/images" )
async def add_image( file: Annotated[ bytes, File() ] ):
    img_hash = sha256( file ).hexdigest()
    
    img_emb = await get_img_emb( img_hash )

    if img_emb:
        raise HTTPException(
            status_code = 409,
            detail = "Image already exists"
        )

    img_emb = np.random.rand( 4096 ).astype( np.float32 ).tobytes()

    await r.hset(
        f"img:{img_hash}",
        mapping = { "emb": img_emb, "url": f"https://cloud.com/images/{img_hash}" }
    )


async def get_img_emb( img_hash ):
    return await r.hget( f"img:{img_hash}", "emb" )

def gen_img_emb( img ):
    img = tf.io.decode_image( img )

    img = img[ tf.newaxis, ... ]

    img = tf.image.resize( img, ( 224, 224 ) )

    img = img[ 0, ... ]

    print( img.shape )

    raise HTTPException(
        status_code = 401,
        detail = "Relax"
    )

query = (
    Query( "*=>[KNN $n @emb $vec as vec_score]" )
    .sort_by( "vec_score" )
    #.return_fields( "vec_score", "emb" )
    .return_fields( "url" )
    .dialect( 4 )
)

@app.post( "/search" )
async def search( n: int, img: Annotated[ bytes, File() ] ):
    img_hash = sha256( img ).hexdigest()

    img_emb = await get_img_emb( img_hash )

    if not img_emb:
        img_emb = gen_img_emb( img )

    aux = await r.ft( "idx:img_emb" ).search(
        query,
        { "n": n, "vec": img_emb }
    )

    urls = []

    for doc in aux.docs:
        urls.append( doc.url )

    return urls
