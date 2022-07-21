import uvicorn

if __name__ == "__main__":
    print("Starting API server on port")
    uvicorn.run("endpoints:app", port=8822, reload=True, access_log=False)