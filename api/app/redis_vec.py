from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

class VectorClient:
    def __init__( self, r ):
        self._INDEX_NAME = "idx:img_emb"
        self._DOC_PREFIX = "img:"
        self._VEC_SIZE = 4096

        self.r = r

    def create_index( self, host, port ):
        r = Redis( host = host, port = port )

        try:
            r.ft( self._INDEX_NAME ).info()
        except:
            schema = (
                VectorField(
                    "emb",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._VEC_SIZE,
                        "DISTANCE_METRIC": "COSINE",
                    }
                ),
            )

            r.ft( self._INDEX_NAME ).create_index(
                fields = schema,
                definition = IndexDefinition(
                    prefix = [ self._DOC_PREFIX ],
                    index_type = IndexType.HASH
                )
            )

    query = (
        Query( "*=>[KNN $n @emb $vec as vec_score]" )
        .sort_by( "vec_score" )
        .return_fields( "url", "vec_score" )
        .dialect( 4 )
    )

    async def knn_search( r, n, img_emb ):
        aux = await r.ft( "idx:img_emb" ).search(
            query,
            { "n": n, "vec": img_emb }
        )
