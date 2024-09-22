from os import getenv

import asyncio

import redis.asyncio as redis

from fastapi import FastAPI, File, HTTPException

from typing import Annotated

from hashlib import sha256

from redis_vec import VectorClient

from model import build_model, preprocess_img

REDIS_HOST = getenv( "REDIS_HOST" )
REDIS_PORT = getenv( "REDIS_PORT" )
#REDIS_HOST = "localhost"
#REDIS_PORT = "6379"


r = redis.Redis( host = REDIS_HOST, port = REDIS_PORT )

v = VectorClient( r )
v.create_index( REDIS_HOST, REDIS_PORT )

app = FastAPI()

model = build_model()


@app.post( "/images" )
async def add_image( img: Annotated[ bytes, File() ] ):
    img_hash = sha256( img ).hexdigest()
    
    img_emb = await get_img_emb( img_hash )

    if img_emb:
        raise HTTPException(
            status_code = 409,
            detail = "Image already exists"
        )

    img_emb = gen_img_emb( img )

    await r.hset(
        f"img:{img_hash}",
        mapping = { "emb": img_emb, "url": f"https://cloud.com/images/{img_hash}" }
    )


async def get_img_emb( img_hash ):
    return await r.hget( f"img:{img_hash}", "emb" )

def gen_img_emb( img ):
    emb = model.predict( preprocess_img( img ) )

    emb = bytes( emb )

    return emb

@app.post( "/search" )
async def search( n: int, img: Annotated[ bytes, File() ] ):
    img_hash = sha256( img ).hexdigest()

    img_emb = await get_img_emb( img_hash )

    if not img_emb:
        img_emb = gen_img_emb( img )

    res = v.knn_search( n, img_emb )

    urls = []

    for doc in res.docs:
        urls.append( { "url": doc.url, "score": doc.vec_score } )

    return urls
