import logging
import time
from typing import Optional, Any, Dict
import redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    ConnectionError,
    TimeoutError,
    BusyLoadingError
)

class RedisStreamManager:
    """
    Manages the lifecycle of a Redis Stream Consumer Group.
    Ensures connectivity, group creation, and reliable message acknowledgement.
    """
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 stream_key: str = 'price_data_stream',
                 group_name: str = 'trading_bot_group',
                 dlq_key: str = 'market_ticks_dlq',
                 password: Optional[str] = None):
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.stream_key = stream_key
        self.group_name = group_name
        self.dlq_key = dlq_key
        
        self.logger = logging.getLogger("RedisStreamManager")
        self._initialize()

    def _initialize(self) -> None:
        """
        Sets up the Redis Connection Pool with robust Retry logic.
        Uses Exponential Backoff to handle temporary network failures.
        """
        # Cap wait time at 10 seconds, start at 1 second
        backoff_strategy = ExponentialBackoff(cap=10.0, base=1.0)
        
        # Retry up to 10 times on specific network errors
        retry_logic = Retry(
            backoff=backoff_strategy,
            retries=10
        )
        
        self.pool = redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )
        
        self.r = redis.Redis(
            connection_pool=self.pool,
            retry=retry_logic,
            retry_on_error=[
                ConnectionError,
                TimeoutError,
                ConnectionResetError, # Using Python's built-in exception
                BusyLoadingError
            ],
            health_check_interval=1,
            socket_timeout=5,
            socket_connect_timeout=15
        )

    def ensure_group(self) -> None:
        """
        Idempotently creates the Consumer Group.
        Ignores error if group already exists (BUSYGROUP).
        """
        try:
            # '0' indicates reading from the beginning, mkstream=True creates stream if missing
            self.r.xgroup_create(self.stream_key, self.group_name, id='0', mkstream=True)
            self.logger.info(f"Consumer group '{self.group_name}' ready on stream '{self.stream_key}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                self.logger.debug(f"Consumer group '{self.group_name}' already exists.")
            else:
                self.logger.critical(f"Critical Redis Error during group creation: {e}")
                raise e
        except Exception as e:
            self.logger.error(f"Unexpected error ensuring group: {e}")

    def add_event(self, stream_key: str, data: Dict[str, Any], maxlen: int = 5000) -> str:
        """
        Helper to safely add events to a stream with strict memory capping.
        Prevents Redis OOM crashes by enforcing maxlen (Default 5000).
        """
        try:
            return self.r.xadd(stream_key, data, maxlen=maxlen, approximate=True)
        except Exception as e:
            self.logger.error(f"Failed to add event to stream {stream_key}: {e}")
            return ""

    def trim_stream(self, maxlen: int = 5000) -> None:
        """
        Trims the stream to a fixed size to prevent memory leaks.
        Default hardened to 5000.
        """
        try:
            self.r.xtrim(self.stream_key, maxlen=maxlen, approximate=True)
        except Exception as e:
            self.logger.error(f"Failed to trim stream: {e}")

def get_redis_connection(host: str = 'localhost',
                         port: int = 6379,
                         db: int = 0,
                         password: Optional[str] = None,
                         decode_responses: bool = True) -> redis.Redis:
    """
    Factory function for a standalone Redis connection with Retry logic.
    Used by Shared Utilities, Config, and Dashboard.
    """
    backoff_strategy = ExponentialBackoff(cap=10.0, base=1.0)
    
    retry_logic = Retry(
        backoff=backoff_strategy,
        retries=10
    )
    
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        decode_responses=decode_responses,
        retry=retry_logic,
        retry_on_error=[
            ConnectionError,
            TimeoutError,
            ConnectionResetError, # Using Python's built-in exception
            BusyLoadingError
        ],
        health_check_interval=1,
        socket_timeout=5,
        socket_connect_timeout=15
    )