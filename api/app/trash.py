import redis
from redis.commands.search.query import Query
import numpy as np

r = redis.Redis()

p = r.pipeline()
for i in range( 100 ):
    a = np.random.rand( 4 )
    p.hset( f"img:{i}", "emb", a.astype( np.float32 ).tobytes() )
p.execute()

query = (
    Query( "*=>[KNN $n @emb $vec as vec_score]" )
    .sort_by( "vec_score" )
    .return_fields( "vec_score", "emb" )
    .dialect( 4 )
)

aux = r.ft( "idx:img_emb" ).search(
    query,
    { "n": 3, "vec": np.random.rand( 4 ).astype( np.float32 ).tobytes() }
).docs

print( aux )
