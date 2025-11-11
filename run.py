#!/usr/bin/env python3
"""Simple server runner without uvicorn CLI."""

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Roll Analyzer API...")
    print("📍 Server will be available at: http://localhost:8015")
    print("📚 API docs at: http://localhost:8015/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8015,
        reload=True,
        log_level="info"
    )


