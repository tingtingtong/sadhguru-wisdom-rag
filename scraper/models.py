from pydantic import BaseModel


class ScrapedDocument(BaseModel):
    title: str
    url: str
    content: str
    content_type: str = "article"
    topics: list[str] = []
    date: str = ""
    excerpt: str = ""
