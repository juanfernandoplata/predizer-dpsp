import redis
from redis.commands.search.query import Query
import numpy as np
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

r = redis.Redis()

def create_index():
    INDEX_NAME = "idx:img_emb"
    DOC_PREFIX = "img:"

    try:
        r.ft( INDEX_NAME ).info()
        print("Index already exists!")
    except:
        schema = (
            VectorField(
                "emb",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 4096,
                    "DISTANCE_METRIC": "COSINE",
                }
            ),
        )

        r.ft( INDEX_NAME ).create_index(
            fields = schema,
            definition = IndexDefinition(
                prefix = [ DOC_PREFIX ],
                index_type=IndexType.HASH
            )
        )

create_index()

#p = r.pipeline()
#for i in range( 100 ):
#    a = np.random.rand( 4 )
#    p.hset( f"img:{i}", "emb", a.astype( np.float32 ).tobytes() )
#p.execute()
#
#query = (
#    Query( "*=>[KNN $n @emb $vec as vec_score]" )
#    .sort_by( "vec_score" )
#    .return_fields( "vec_score", "emb" )
#    .dialect( 4 )
#)
#
#aux = r.ft( "idx:img_emb" ).search(
#    query,
#    { "n": 3, "vec": np.random.rand( 4 ).astype( np.float32 ).tobytes() }
#).docs
#
#print( aux )
