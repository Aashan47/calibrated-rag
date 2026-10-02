# API rate limits and webhooks

The REST API is rate-limited per token: 300 requests per minute on the Team plan, 1,000 per minute on Business and 5,000 per minute on Enterprise. The Free plan does not include API access. When you exceed the limit the API returns HTTP 429 with a Retry-After header; back off for that many seconds. Webhook deliveries that fail (any non-2xx response) are retried 5 times with exponential backoff over 24 hours, after which the event is dropped and the webhook is marked as failing in Settings › Integrations › Webhooks.
