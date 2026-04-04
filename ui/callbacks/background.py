from dash import CeleryManager, DiskcacheManager
from core.config import REDIS_URL

def background_manager():
    if REDIS_URL:
        # Use Redis & Celery if REDIS_URL set as an env variable
        from celery import Celery
        celery_app = Celery(__name__, broker=REDIS_URL, backend=REDIS_URL)
        background_callback_manager = CeleryManager(celery_app)
    else:
        # Diskcache for non-production apps when developing locally
        import diskcache
        cache = diskcache.Cache("./cache")
        background_callback_manager = DiskcacheManager(cache)
    
    return background_callback_manager