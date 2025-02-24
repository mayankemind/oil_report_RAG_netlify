# PlotSemanticSearch
{
    "mappings": {
        "dynamic": true,
        "fields": {
            "embedding": {
                "dimensions": 384,
                "similarity": "dotProduct",
                "type": "knnVector"
            }
        }
    }
}
