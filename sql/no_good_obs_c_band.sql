\copy (
SELECT t.object_name, t.obs_date, t.freq, t.file_name, t.comment
FROM maps_apr25_test AS t
WHERE t.freq BETWEEN 2.8e9 AND 7e9
  AND NOT EXISTS (
    SELECT 1
    FROM maps_apr25_test AS x
    WHERE x.object_name = t.object_name
      AND x.freq BETWEEN 2.8e9 AND 7e9
      AND x.map_quality = 1
  )
) to 'csv/no_good_obs_c_band.csv' with csv header;

-- \copy (
-- SELECT t.object_name, t.obs_date, t.freq, t.file_name, t.comment
-- FROM maps_apr25_test_upd AS t
-- WHERE t.freq BETWEEN 2.8e9 AND 7e9
--   AND NOT EXISTS (
--     SELECT 1
--     FROM maps_apr25_test_upd AS x
--     WHERE x.object_name = t.object_name
--       AND x.freq BETWEEN 2.8e9 AND 7e9
--       AND x.map_quality = 1
--   )
-- ) to 'csv/no_good_obs_c_band_upd.csv' with csv header;