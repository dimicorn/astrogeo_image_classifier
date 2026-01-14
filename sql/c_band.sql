-- Interactive mode only
\copy (
  select object_name, obs_date, freq, file_name, comment
  from maps_apr25_new
  where map_quality = 0 and freq >= 2.8e9 and freq <= 7e9
) to 'csv/maps_apr25_new_c_band.csv' with csv header;

-- \copy (
--   select object_name, obs_date, freq, file_name, comment
--   from maps_apr25_original
--   where map_quality = 0 and freq >= 2.8e9 and freq <= 7e9
-- ) to 'csv/maps_apr25_orig_c_band.csv' with csv header;

-- \copy (
--   WITH t AS (
--     SELECT
--       object_name,
--       obs_date::text AS obs_date,
--       to_char(freq, 'FM9999999999') AS freq_fmt,
--       file_name,

--       CASE
--         WHEN comment ILIKE '%distance%' THEN
--           'distance=' ||
--           to_char(
--             substring(comment from '([0-9]+(\.[0-9]+)?)')::double precision,
--             'FM999999.0'
--           )

--         WHEN comment ILIKE '%snr%' THEN
--           'snr=' ||
--           to_char(
--             substring(comment from '([0-9]+(\.[0-9]+)?)')::double precision,
--             'FM999999.0'
--           )

--         ELSE comment
--       END AS comment_fmt

--     FROM maps_apr25_original
--     WHERE map_quality = 0
--       AND freq BETWEEN 2.8e9 AND 7e9
--   )
--   SELECT format(
--       '%-15s %-12s %-12s %-45s %s',
--       object_name,
--       obs_date,
--       freq_fmt,
--       file_name,
--       comment_fmt
--     ) AS line
--   FROM t
--   ORDER BY object_name, obs_date
-- ) TO 'csv/maps_apr25_orig_c_band.txt' WITH (FORMAT text);