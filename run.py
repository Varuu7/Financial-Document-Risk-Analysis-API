import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    print("=" * 65)
    print(f"  {settings.app_name}")
    print(f"  Swagger UI Documentation: http://localhost:{settings.port}/docs")
    print(f"  ReDoc Documentation:      http://localhost:{settings.port}/redoc")
    print(f"  OpenAPI Schema:           http://localhost:{settings.port}/openapi.json")
    print("=" * 65)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
