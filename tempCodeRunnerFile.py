from fastapi import FastAPI, Path, Query
from typing import Annotated

app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(item_id: int = Path(title="The ID of the item to get"), q: str = Query(alias="item-query")):
    pass