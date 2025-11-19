\copy (
SELECT t.object_name, t.obs_date, t.freq, t.file_name, t.comment
FROM maps_apr25_new AS t
WHERE t.freq BETWEEN 2.8e9 AND 7e9
  AND NOT EXISTS (
    SELECT 1
    FROM maps_apr25_new AS x
    WHERE x.object_name = t.object_name
      AND x.freq BETWEEN 2.8e9 AND 7e9
      AND x.map_quality = 1
  )
) to 'csv/no_good_obs_new_c_band.csv' with csv header;

\copy (
SELECT t.object_name, t.obs_date, t.freq, t.file_name, t.comment
FROM maps_apr25_original AS t
WHERE t.freq BETWEEN 2.8e9 AND 7e9
  AND NOT EXISTS (
    SELECT 1
    FROM maps_apr25_original AS x
    WHERE x.object_name = t.object_name
      AND x.freq BETWEEN 2.8e9 AND 7e9
      AND x.map_quality = 1
  )
) to 'csv/no_good_obs_orig_c_band.csv' with csv header;