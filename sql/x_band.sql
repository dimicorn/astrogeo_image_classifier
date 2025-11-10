-- Interactive mode only
\copy (
  select object_name, obs_date, freq, file_name, comment
  from maps_apr25_test
  where map_quality = 0 and freq >= 7e9 and freq <= 9e9
) to 'maps_apr25_test_x_band.csv' with csv header;

-- \copy (
--   select object_name, obs_date, freq, file_name, comment
--   from maps_apr25_test_upd
--   where map_quality = 0 and freq >= 7e9 and freq <= 9e9
-- ) to 'maps_apr25_test_upd_x_band.csv' with csv header;