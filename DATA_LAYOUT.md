# Expected data layout inferred from the uploaded `src.zip`

```text
data/
  num_collection/domestic/raw/                 # original numeric CSV/Excel tables
  text_collection/raw/macro/global/            # macro JSONL by month
  text_collection/raw/micro/kamakura/          # Kamakura micro JSONL
  text_collection/raw/micro/yokohama/          # Yokohama micro JSONL
  text_collection/scored/{city}/               # TQSM-scored records
  text_collection/topk/{city}/{variant}/{mode}/ # monthly selections

data_Yelp/
  num_collection/MSY_airport_arriving_2015_2025_cleaned.csv
  text_collection/new_orleans_tourism_reviews_2015_2025_monthly_top200_cleaned.csv
hotelrec_data/
  num_collection/Amsterdam_hotel_arriving_tourists_2012_202602_cleaned.csv
  text_collection/hotelrec_amsterdam_2012plus_monthly_date_text_top200.csv
```

Domestic numeric reader expects Japanese columns, notably `都道府県名`, `地域名称`, `年`, `月`, `人数` with correct files under `data/num_collection/domestic/raw`.

Notes: The paths above are from `Preview`, `simple_monthly_loader`, `scoring_utils`, `text_handler`, not from checking actual data because **data was not supplied**. Verify the 3 copied datasets before publication. `run_rescore_existing.py` defaults to `scored_trend`, whereas Top-k defaults to `scored`: reconcile the run config/file placement without overwriting data. The paper calls for 5 macro plus 5 micro selected texts per month; `TopKGenerator(...,k=10,mode='balanced')` splits this into 5 and 5.
