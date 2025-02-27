import uvicorn
from fastapi import FastAPI

from app import Environment
from app.core.logging import configure_logging
from app.utils.advanced_logger import AdvancedLogger

configure_logging()

app = FastAPI(title=Environment.APP_TITLE, debug=Environment.DEBUG)

uvicorn.run(app, host=Environment.HOST, port=Environment.PORT, log_config=None)
