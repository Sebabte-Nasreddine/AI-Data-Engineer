{{ config(schema='ai', materialized='table', tags=['ai']) }}

with reviews as (
    select
        review_id,
        restaurant_id,
        rating,
        comment as text,
        review_date,
        city
    from {{ ref('stg_reviews') }}
    where length(trim(comment)) >= 15   -- élimine "good", "ok", bruit sémantique
),

enriched as (
    select review_id, sentiment_label, topic, key_issue
    from {{ source('ai', 'review_enriched') }}
),

restaurants as (
    select restaurant_id, restaurant_name, cuisine
    from {{ ref('dim_restaurants') }}
)

select
    r.review_id,
    r.text,
    r.city,
    rest.cuisine,
    r.restaurant_id,
    rest.restaurant_name,
    r.rating,
    r.review_date,
    coalesce(e.sentiment_label, 'unknown') as sentiment_label,
    coalesce(e.topic, 'unknown')           as topic,
    e.key_issue,
    md5(r.text)                            as content_hash
from reviews r
left join restaurants  rest on rest.restaurant_id = r.restaurant_id
left join enriched     e    on e.review_id        = r.review_id