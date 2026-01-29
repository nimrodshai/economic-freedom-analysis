# PostHog Analysis Queries

## Bot Detection Setup

### Known Bot Locations (Data Centers)
```
Council Bluffs, Boardman, Ashburn, Colorado Springs, Beauharnois,
Frankfurt am Main, Amsterdam, Las Vegas, Prague, Santa Clara
```

### Bot Filter IP Prefixes
```
35.192.0.0/12,35.208.0.0/12,34.64.0.0/10,35.224.0.0/12,34.0.0.0/9,13.64.0.0/11,52.0.0.0/11,54.64.0.0/12
```

### Bot Filter User Agent Patterns
```
HeadlessChrome,Headless,PhantomJS,Lighthouse,PageSpeed,GTmetrix,Pingdom,UptimeRobot,SemrushBot,AhrefsBot,DotBot,MJ12bot,BLEXBot,PetalBot,DataForSeoBot,serpstatbot,Screaming Frog,Bytespider,GPTBot,ChatGPT,ClaudeBot,anthropic-ai,CCBot
```

### Bot Signatures
- Chrome on Linux from data center IPs
- Single pageview bounces (no $pageleave event)
- Time on page < 5 seconds
- Screen size 800x600 (headless browser default)

---

## Useful HogQL Queries

### Real Users Only - Daily Traffic
```sql
SELECT
    toDate(timestamp) as date,
    count(*) as pageviews,
    count(DISTINCT person_id) as unique_visitors
FROM events
WHERE event = '$pageview'
    AND timestamp >= now() - INTERVAL 7 DAY
    AND properties['$geoip_city_name'] NOT IN (
        'Council Bluffs', 'Boardman', 'Ashburn', 'Colorado Springs',
        'Beauharnois', 'Frankfurt am Main', 'Amsterdam', 'Las Vegas',
        'Prague', 'Santa Clara'
    )
    AND NOT (properties['$browser'] = 'Chrome' AND properties['$os'] = 'Linux')
GROUP BY date
ORDER BY date
```

### Real Users by Country
```sql
SELECT
    properties['$geoip_country_name'] AS country,
    count(DISTINCT person_id) as unique_visitors,
    count(*) as pageviews
FROM events
WHERE event = '$pageview'
    AND timestamp >= now() - INTERVAL 7 DAY
    AND properties['$geoip_city_name'] NOT IN (
        'Council Bluffs', 'Boardman', 'Ashburn', 'Colorado Springs',
        'Beauharnois', 'Frankfurt am Main', 'Amsterdam', 'Las Vegas',
        'Prague', 'Santa Clara'
    )
    AND NOT (properties['$browser'] = 'Chrome' AND properties['$os'] = 'Linux')
GROUP BY country
ORDER BY unique_visitors DESC
```

### Time on Page Analysis (Bot Detection)
```sql
SELECT
    properties['$geoip_country_name'] AS country,
    properties['$geoip_city_name'] AS city,
    properties['$prev_pageview_duration'] AS seconds_on_page,
    properties['$browser'] AS browser,
    properties['$os'] AS os,
    timestamp
FROM events
WHERE event = '$pageleave'
    AND timestamp >= now() - INTERVAL 7 DAY
ORDER BY timestamp DESC
LIMIT 50
```

### Raw Event Details (for investigation)
```sql
SELECT
    properties['$geoip_country_name'] AS country,
    properties['$geoip_city_name'] AS city,
    properties['$browser'] AS browser,
    properties['$os'] AS os,
    properties['$referring_domain'] AS referrer,
    timestamp
FROM events
WHERE event = '$pageview'
    AND timestamp >= now() - INTERVAL 24 HOUR
ORDER BY timestamp DESC
LIMIT 50
```

### Traffic Sources
```sql
SELECT
    properties['$referring_domain'] AS referrer,
    count(DISTINCT person_id) as unique_visitors
FROM events
WHERE event = '$pageview'
    AND timestamp >= now() - INTERVAL 7 DAY
    AND properties['$geoip_city_name'] NOT IN (
        'Council Bluffs', 'Boardman', 'Ashburn', 'Colorado Springs',
        'Beauharnois', 'Frankfurt am Main', 'Amsterdam', 'Las Vegas',
        'Prague', 'Santa Clara'
    )
GROUP BY referrer
ORDER BY unique_visitors DESC
```

### Device Breakdown
```sql
SELECT
    properties['$device_type'] AS device,
    count(DISTINCT person_id) as unique_visitors
FROM events
WHERE event = '$pageview'
    AND timestamp >= now() - INTERVAL 7 DAY
    AND properties['$geoip_city_name'] NOT IN (
        'Council Bluffs', 'Boardman', 'Ashburn', 'Colorado Springs',
        'Beauharnois', 'Frankfurt am Main', 'Amsterdam', 'Las Vegas',
        'Prague', 'Santa Clara'
    )
GROUP BY device
ORDER BY unique_visitors DESC
```

---

## Quick Bot vs Real Check

A visitor is likely a **bot** if:
- City is a known data center location
- Browser is Chrome + OS is Linux
- Time on page < 5 seconds
- Screen size is exactly 800x600
- No referrer ($direct) + single pageview + instant bounce

A visitor is likely **real** if:
- Time on page > 30 seconds
- Multiple pageviews in session
- Has $pageleave event with reasonable duration
- Normal screen sizes (varies)
- Mobile Safari on iOS (hard to spoof)
