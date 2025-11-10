-- Interactive mode only
\copy (
  select object_name, obs_date, freq, file_name, comment
  from maps_apr25_test
  where map_quality = 0 and freq >= 1.8e9 and freq <= 2.8e9
) to 'csv/maps_apr25_test_s_band.csv' with csv header;

-- \copy (
--   select object_name, obs_date, freq, file_name, comment
--   from maps_apr25_test_upd
--   where map_quality = 0 and freq >= 1.8e9 and freq <= 2.8e9
-- ) to 'csv/maps_apr25_test_upd_s_band.csv' with csv header;